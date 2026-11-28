# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline sizing contracts: architecture counts, memory arithmetic and fits."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from eyvan.capacity.providers.gcp.tpu_types import parse_tpu_accelerator_type
from eyvan.domain.sizing import (
    DEFAULT_CANDIDATE_TYPES,
    DEFAULT_HEADROOM,
    GIB,
    TPU_HBM_GIB,
    MemoryEstimate,
    ModelShape,
    SizingRequest,
    SliceFit,
    estimate_memory,
    fitting_slices,
    model_shape_from_config,
    shape_from_parameter_count,
    size_job,
    smallest_fit,
)


def _llama_config() -> dict[str, object]:
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "tie_word_embeddings": False,
    }


def _seven_billion() -> ModelShape:
    return model_shape_from_config(_llama_config())


def _memory(total_bytes: int) -> MemoryEstimate:
    return MemoryEstimate(total_bytes, 0, 0, 0, 0, 0, total_bytes, ())


def test_llama_two_seven_billion_parameter_count() -> None:
    shape = _seven_billion()
    assert shape.parameters == 6_738_415_616
    assert shape.parameters == pytest.approx(6.738e9, rel=0.005)
    assert shape.active_parameters == shape.parameters
    assert (shape.num_kv_heads, shape.head_dim) == (32, 128)


def test_llama_three_eight_billion_parameter_count() -> None:
    config = _llama_config() | {"intermediate_size": 14336, "vocab_size": 128256, "num_key_value_heads": 8}
    shape = model_shape_from_config(config)
    assert shape.parameters == 8_030_261_248
    assert shape.parameters == pytest.approx(8.030e9, rel=0.005)


def test_tied_embeddings_remove_exactly_one_output_head() -> None:
    shape = model_shape_from_config(_llama_config() | {"tie_word_embeddings": True})
    assert shape.parameters == _seven_billion().parameters - 32000 * 4096


def test_mixtral_total_and_active_counts_are_different() -> None:
    config = _llama_config() | {
        "model_type": "mixtral",
        "intermediate_size": 14336,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }
    shape = model_shape_from_config(config)
    assert shape.parameters == 46_702_792_704
    assert shape.parameters == pytest.approx(46.7e9, rel=0.01)
    assert shape.active_parameters == pytest.approx(12.9e9, rel=0.01)
    estimate = estimate_memory(SizingRequest(shape))
    assert estimate.parameters_bytes == shape.parameters * 2
    assert any("All MoE experts" in note for note in estimate.notes)


def test_qwen_moe_expert_alias_and_intermediate_size() -> None:
    config = _llama_config() | {
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 14336,
        "shared_expert_intermediate_size": 1024,
    }
    shape = model_shape_from_config(config)
    without_shared = model_shape_from_config(config | {"shared_expert_intermediate_size": 0})
    shared_parameters = 32 * (3 * 4096 * 1024 + 4096)
    assert shape.parameters == without_shared.parameters + shared_parameters
    assert shape.active_parameters == without_shared.active_parameters + shared_parameters


def test_sparse_moe_layers_keep_dense_mlp_layers() -> None:
    config = _llama_config() | {
        "num_hidden_layers": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 1024,
    }
    dense = model_shape_from_config(_llama_config() | {"num_hidden_layers": 4})
    sparse = model_shape_from_config(config | {"decoder_sparse_step": 2, "mlp_only_layers": [1]})
    # Only layer 3 is sparse: all other layers retain the dense MLP.
    dense_mlp = 3 * 4096 * 11008
    expert_mlp = 3 * 4096 * 1024
    assert sparse.parameters == dense.parameters - dense_mlp + expert_mlp * 8 + 4096 * 8


def test_explicit_gemma_head_dimension_and_default_kv_heads() -> None:
    config = _llama_config() | {"head_dim": 256}
    del config["num_key_value_heads"]
    shape = model_shape_from_config(config)
    assert shape.num_kv_heads == 32
    assert shape.head_dim == 256
    assert shape.parameters > _seven_billion().parameters


def test_missing_architecture_fields_fail_clearly() -> None:
    for name in ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads", "intermediate_size"):
        config = _llama_config()
        del config[name]
        with pytest.raises(ValueError, match=name):
            model_shape_from_config(config)


def test_invalid_architecture_dimensions_are_rejected() -> None:
    for values in (
        {"hidden_size": 0},
        {"num_attention_heads": True},
        {"num_attention_heads": 31},
        {"num_key_value_heads": 7},
        {"tie_word_embeddings": "false"},
        {"num_local_experts": 8},
        {"num_local_experts": 8, "num_experts_per_tok": 9},
    ):
        with pytest.raises(ValueError):
            model_shape_from_config(_llama_config() | values)


def test_parameter_only_shape_has_documented_defaults_and_overrides() -> None:
    shape = shape_from_parameter_count(7_000_000_000)
    assert (shape.hidden_size, shape.num_layers, shape.vocab_size) == (4096, 32, 32000)
    assert shape.active_parameters == shape.parameters
    custom = shape_from_parameter_count(100_000, hidden_size=128, num_layers=2, vocab_size=256)
    assert (custom.hidden_size, custom.num_layers, custom.vocab_size) == (128, 2, 256)
    assert any("rough defaults" in note for note in estimate_memory(SizingRequest(shape)).notes)


def test_frozen_data_classes_and_shape_validation() -> None:
    shape = _seven_billion()
    with pytest.raises(FrozenInstanceError):
        shape.parameters = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="active_parameters"):
        replace(shape, active_parameters=shape.parameters + 1)
    for value in (0, -1, True):
        with pytest.raises(ValueError):
            shape_from_parameter_count(value)


def test_inference_is_smaller_than_sft_is_smaller_than_dpo() -> None:
    request = SizingRequest(_seven_billion())
    inference = estimate_memory(replace(request, trainer="inference"))
    sft = estimate_memory(request)
    dpo = estimate_memory(replace(request, trainer="dpo"))
    assert inference.total_bytes < sft.total_bytes < dpo.total_bytes
    assert inference.gradients_bytes == inference.optimizer_bytes == 0
    assert dpo.reference_bytes == sft.parameters_bytes
    assert sft.gradients_bytes == sft.parameters_bytes
    assert sft.optimizer_bytes == request.shape.parameters * 12


def test_optimizer_state_and_master_weight_arithmetic() -> None:
    shape = shape_from_parameter_count(1000, hidden_size=16, num_layers=2, vocab_size=32)
    request = SizingRequest(shape)
    for optimizer, bf16_bytes, fp32_bytes in (
        ("adamw", 12000, 8000),
        ("lion", 8000, 4000),
        ("adafactor", 4400, 400),
        ("sgd", 4000, 0),
        ("none", 0, 0),
    ):
        assert estimate_memory(replace(request, optimizer=optimizer)).optimizer_bytes == bf16_bytes
        fp32 = estimate_memory(replace(request, optimizer=optimizer, dtype="float32"))
        assert fp32.optimizer_bytes == fp32_bytes
        assert fp32.parameters_bytes == 4000
    assert estimate_memory(replace(request, dtype="float16")) == estimate_memory(request)


def test_lora_uses_adapter_gradients_and_much_less_optimizer_memory() -> None:
    request = SizingRequest(_seven_billion())
    full = estimate_memory(request)
    lora = estimate_memory(replace(request, lora_rank=16))
    adapters = 2 * 16 * 4096 * 7 * 32
    assert lora.optimizer_bytes == adapters * 12
    assert lora.optimizer_bytes < full.optimizer_bytes / 50
    assert lora.gradients_bytes == adapters * 2
    assert lora.parameters_bytes == full.parameters_bytes + adapters * 2


def test_orpo_has_no_reference_and_grpo_does() -> None:
    request = SizingRequest(_seven_billion())
    orpo = estimate_memory(replace(request, trainer="orpo"))
    grpo = estimate_memory(replace(request, trainer="grpo"))
    assert orpo.reference_bytes == 0
    assert grpo.reference_bytes == request.shape.parameters * 2
    assert any("reference-free" in note for note in orpo.notes)


def test_checkpointing_reduces_activation_memory() -> None:
    request = SizingRequest(_seven_billion())
    checkpointed = estimate_memory(request)
    normal = estimate_memory(replace(request, gradient_checkpointing=False))
    assert checkpointed.activations_bytes == 8 * 2048 * 4096 * 32 * 2 * 2
    assert normal.activations_bytes == checkpointed.activations_bytes * 8


def test_grpo_generation_cache_accounts_for_gqa_output_length_and_generations() -> None:
    shape = model_shape_from_config(_llama_config() | {"num_key_value_heads": 8})
    request = SizingRequest(shape, trainer="grpo", generation_tokens=512, num_generations=4)
    estimate = estimate_memory(request)
    assert estimate.generation_bytes == 2 * 32 * 8 * 128 * 2 * (2048 + 512) * 8 * 4
    assert estimate.generation_bytes > 0
    assert estimate.generation_bytes > estimate_memory(replace(request, generation_tokens=0)).generation_bytes
    assert estimate.generation_bytes == estimate_memory(replace(request, num_generations=1)).generation_bytes * 4


def test_missing_kv_dimensions_use_mha_width_for_inference_cache() -> None:
    shape = shape_from_parameter_count(7_000_000_000)
    estimate = estimate_memory(SizingRequest(shape, trainer="inference"))
    assert estimate.generation_bytes == 2 * 32 * 4096 * 2 * 2048 * 8


def test_memory_components_sum_to_total() -> None:
    estimate = estimate_memory(SizingRequest(_seven_billion(), trainer="grpo", generation_tokens=128))
    assert estimate.total_bytes == sum(
        (
            estimate.parameters_bytes,
            estimate.gradients_bytes,
            estimate.optimizer_bytes,
            estimate.reference_bytes,
            estimate.activations_bytes,
            estimate.generation_bytes,
        )
    )
    with pytest.raises(ValueError, match="sum"):
        replace(estimate, total_bytes=0)


def test_invalid_dtype_trainer_and_optimizer_raise_value_error() -> None:
    request = SizingRequest(_seven_billion())
    with pytest.raises(ValueError, match="dtype"):
        replace(request, dtype="int8")
    with pytest.raises(ValueError, match="trainer"):
        replace(request, trainer="unknown")
    with pytest.raises(ValueError, match="optimizer"):
        replace(request, optimizer="adamm")


def test_invalid_request_dimensions_are_rejected() -> None:
    request = SizingRequest(_seven_billion())
    for field in ("sequence_length", "global_batch_size", "num_generations", "lora_rank"):
        with pytest.raises(ValueError, match=field):
            replace(request, **{field: 0})
    with pytest.raises(ValueError, match="generation_tokens"):
        replace(request, generation_tokens=-1)
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        replace(request, gradient_checkpointing="false")  # type: ignore[arg-type]


def test_seven_billion_training_needs_more_than_one_litepod_chip_but_fits_v5p() -> None:
    estimate = estimate_memory(SizingRequest(shape_from_parameter_count(7_000_000_000)))
    litepod = fitting_slices(estimate, ("v5litepod-1",))[0]
    assert not litepod.fits
    fit = smallest_fit(estimate, ("v5p-8", "v5p-16", "v5p-32"))
    assert fit is not None
    assert fit.accelerator_type == "v5p-8"
    assert fit.chips == 4


def test_tpu_aliases_use_physical_chip_counts_not_core_suffixes() -> None:
    fits = fitting_slices(_memory(GIB), ("v4-8", "v5p-8", "v5e-8", "v5litepod-8", "v6e-8"))
    by_type = {fit.accelerator_type: fit for fit in fits}
    assert {name: fit.chips for name, fit in by_type.items()} == {
        "v4-8": 4,
        "v5p-8": 4,
        "v5e-8": 8,
        "v5litepod-8": 8,
        "v6e-8": 8,
    }
    assert by_type["v5e-8"].hbm_per_chip_gib == by_type["v5litepod-8"].hbm_per_chip_gib == 16
    assert TPU_HBM_GIB["v5p"] == 95


def test_smallest_fit_selects_fewest_chips_and_honors_restrictions() -> None:
    estimate = _memory(80 * GIB)
    candidates = ("v5litepod-16", "v5p-8", "v4-8", "v6e-4")
    fits = fitting_slices(estimate, candidates)
    assert [(fit.chips, fit.accelerator_type) for fit in fits] == sorted(
        (fit.chips, fit.accelerator_type) for fit in fits
    )
    assert smallest_fit(estimate, candidates) == fits[0]
    assert smallest_fit(estimate, ("v5litepod-16",)).accelerator_type == "v5litepod-16"
    assert smallest_fit(estimate, ("v5litepod-1",)) is None
    assert smallest_fit(estimate, ()) is None


def test_headroom_boundary_and_utilization_are_exact() -> None:
    estimate = _memory(16 * GIB)
    fit = fitting_slices(estimate, ("v5litepod-1",), headroom=1)[0]
    assert fit.fits
    assert fit.utilization == 1
    assert fit.per_chip_bytes == estimate.total_bytes
    assert not fitting_slices(estimate, ("v5litepod-1",))[0].fits
    boundary = _memory(8 * GIB)
    assert fitting_slices(boundary, ("v5litepod-1",), headroom=0.5)[0].fits
    assert not fitting_slices(boundary, ("v5litepod-1",), headroom=0.49)[0].fits
    for invalid in (0, -1, 1.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="headroom"):
            fitting_slices(estimate, headroom=invalid)


def test_bad_unknown_and_fractional_chip_types_are_skipped() -> None:
    fits = fitting_slices(_memory(GIB), ("bad", "gpu-8", "v9z-4", "v4-1", "v5p-3", "v6e-1"))
    assert [fit.accelerator_type for fit in fits] == ["v6e-1"]


def test_tpu7x_uses_192_gib_per_chip_and_chip_count_suffix() -> None:
    fits = fitting_slices(_memory(150 * GIB), ("tpu7x-4", "tpu7x-8"))
    by_type = {fit.accelerator_type: fit for fit in fits}
    assert by_type["tpu7x-4"].chips == 4
    assert by_type["tpu7x-4"].hbm_per_chip_gib == 192.0
    assert by_type["tpu7x-4"].fits is True
    assert by_type["tpu7x-8"].chips == 8
    assert TPU_HBM_GIB["tpu7x"] == 192.0


def test_default_candidates_are_parseable_and_have_known_hbm() -> None:
    assert len(DEFAULT_CANDIDATE_TYPES) == len(set(DEFAULT_CANDIDATE_TYPES))
    for name in DEFAULT_CANDIDATE_TYPES:
        parsed = parse_tpu_accelerator_type(name)
        assert parsed is not None
        assert parsed.generation in TPU_HBM_GIB
    assert len(fitting_slices(_memory(GIB))) == len(DEFAULT_CANDIDATE_TYPES)
    assert DEFAULT_HEADROOM == 0.85


def test_size_job_is_small_json_ready_and_recommendation_matches_fits() -> None:
    request = SizingRequest(_seven_billion())
    result = size_job(request)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["parameters"] == result["active_parameters"] == request.shape.parameters
    assert 0 < len(result["fits"]) <= 5
    assert result["recommended"] == smallest_fit(estimate_memory(request)).accelerator_type
    assert result["recommended"] == result["fits"][0]["accelerator_type"]
    assert result["estimate"]["total_gib"] == round(estimate_memory(request).total_bytes / GIB, 2)
    assert result["notes"]
    assert size_job(request, ("v5litepod-1",))["recommended"] is None
    assert size_job(request, ("v5litepod-1",))["fits"] == []


def test_public_records_are_frozen_dataclasses() -> None:
    request = SizingRequest(_seven_billion())
    estimate = estimate_memory(request)
    fit = SliceFit("v5p-8", 4, 95, estimate.total_bytes / 4, True, 0.3)
    for record, field, value in ((request, "trainer", "dpo"), (estimate, "total_bytes", 0), (fit, "chips", 8)):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, value)
