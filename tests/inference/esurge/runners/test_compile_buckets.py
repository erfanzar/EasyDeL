# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from easydel.inference.esurge.runners.execution_manager import ExecutionManager


class _SpmdMesh:
    is_mpmd = False


class _MpmdMesh:
    is_mpmd = True


class _Model:
    mesh = _SpmdMesh()


class _MpmdModel:
    mesh = _MpmdMesh()


def test_spmd_compile_prepares_full_bucket_grid():
    manager = ExecutionManager.__new__(ExecutionManager)
    manager.model = _Model()
    manager.use_aot_forward = False
    manager.min_input_pad = 4
    manager._sampler_min_input_pad = 1
    manager._model_num_tokens_paddings = []

    model_step_calls: list[tuple[int, int]] = []
    sampler_calls: list[int] = []

    manager._compile_model_step_variant = lambda **kwargs: model_step_calls.append(
        (int(kwargs["num_tokens"]), int(kwargs["padded_num_reqs"]))
    )
    manager._compile_sampler_variant = lambda **kwargs: sampler_calls.append(int(kwargs["padded_num_reqs"]))

    ExecutionManager.compile(
        manager,
        num_tokens_paddings=[4, 8, 16],
        num_reqs_max_model_len=8,
        max_pages_per_req=1,
        max_num_reqs=8,
        metadata=object(),
        num_reqs_paddings=[4, 8],
        prune_infeasible_pairs=True,
    )

    assert model_step_calls == [(4, 4), (8, 4), (8, 8), (16, 4), (16, 8)]
    assert sampler_calls == [1, 2, 4, 8]
    assert manager._model_num_tokens_paddings == [4, 8, 16]


def test_mpmd_compile_prepares_fused_split_and_sampler_buckets():
    manager = ExecutionManager.__new__(ExecutionManager)
    manager.model = _MpmdModel()
    manager.use_aot_forward = False
    manager.min_input_pad = 4
    manager._sampler_min_input_pad = 1
    manager._model_num_tokens_paddings = []

    fused_calls: list[tuple[int, int]] = []
    backbone_calls: list[tuple[int, bool]] = []
    lm_head_calls: list[int] = []
    sampler_calls: list[int] = []

    class _ModelExecutor:
        supports_pipeline_model_step = True
        pipeline_runtime = object()

    manager._model_executor = _ModelExecutor()
    manager._compile_pipeline_model_step_variant = lambda **kwargs: fused_calls.append(
        (int(kwargs["num_tokens"]), int(kwargs["padded_num_reqs"]))
    )
    manager._compile_backbone_variant = lambda **kwargs: backbone_calls.append(
        (int(kwargs["num_tokens"]), bool(kwargs["use_pipeline_runtime"]))
    )
    manager._compile_lm_head_variant = lambda **kwargs: lm_head_calls.append(int(kwargs["padded_num_reqs"]))
    manager._compile_sampler_variant = lambda **kwargs: sampler_calls.append(int(kwargs["padded_num_reqs"]))

    ExecutionManager.compile(
        manager,
        num_tokens_paddings=[4, 8, 16],
        num_reqs_max_model_len=8,
        max_pages_per_req=1,
        max_num_reqs=8,
        metadata=object(),
        num_reqs_paddings=[4, 8],
        prune_infeasible_pairs=True,
    )

    assert fused_calls == [(4, 4), (4, 8), (8, 8)]
    assert backbone_calls == [(4, False), (4, True), (8, False), (8, True), (16, False), (16, True)]
    assert lm_head_calls == [4, 8]
    assert sampler_calls == [1, 2, 4, 8]


def test_fused_pp_runtime_logits_bucket_respects_model_min_input_pad():
    manager = ExecutionManager.__new__(ExecutionManager)
    manager.min_input_pad = 4

    assert manager._pipeline_model_logits_bucket(padded_num_reqs=4, sampler_padded_num_reqs=1) == 4
    assert manager._pipeline_model_logits_bucket(padded_num_reqs=8, sampler_padded_num_reqs=2) == 4
    assert manager._pipeline_model_logits_bucket(padded_num_reqs=8, sampler_padded_num_reqs=8) == 8


def test_window_emits_token_accepts_unpadded_request_vectors():
    assert ExecutionManager._window_emits_token(
        scheduled_full_cpu=np.array([1, 1, 1, 1], dtype=np.int32),
        active_mask_full_cpu=np.array([True, True, True, False], dtype=np.bool_),
        req_num_tokens_full_cpu=np.array([2, 2, 2, 2], dtype=np.int32),
        num_computed_tokens_cpu=np.array([1, 1, 1], dtype=np.int32),
        padded_num_reqs=4,
    )


def test_compact_sampler_window_masks_rows_missing_unpadded_cpu_state():
    manager = ExecutionManager.__new__(ExecutionManager)
    manager._sampler_min_input_pad = 1
    manager.max_num_reqs = 4
    manager._sampler_gather_positions_cpu = np.zeros((4,), dtype=np.int32)
    manager._sampler_sampling_seeds_cpu = np.zeros((4,), dtype=np.int32)
    manager._sampler_scatter_positions_cpu = np.zeros((4,), dtype=np.int32)
    manager._sampler_window_row_indices_cpu = np.zeros((4,), dtype=np.int32)
    manager._sampler_scheduled_cpu = np.zeros((4,), dtype=np.int32)
    manager._sampler_seq_lens_cpu = np.zeros((4,), dtype=np.int32)
    manager._sampler_active_mask_cpu = np.zeros((4,), dtype=np.bool_)
    manager._sampler_temperature_cpu = np.ones((4,), dtype=np.float32)
    manager._sampler_top_p_cpu = np.ones((4,), dtype=np.float32)
    manager._sampler_top_k_cpu = np.zeros((4,), dtype=np.int32)
    manager._sampler_min_p_cpu = np.zeros((4,), dtype=np.float32)
    manager._sampler_frequency_penalties_cpu = np.zeros((4,), dtype=np.float32)
    manager._sampler_presence_penalties_cpu = np.zeros((4,), dtype=np.float32)
    manager._sampler_repetition_penalties_cpu = np.ones((4,), dtype=np.float32)

    sample_count, sampler_padded_num_reqs, total_tokens = manager._prepare_compact_sampler_window(
        padded_num_reqs=4,
        scheduled_full_cpu=np.array([1, 1, 1, 1], dtype=np.int32),
        active_mask_full_cpu=np.array([True, True, True, True], dtype=np.bool_),
        window_row_indices_cpu=np.array([10, 11, 12], dtype=np.int32),
        num_computed_tokens_cpu=np.array([4, 5, 6], dtype=np.int32),
        temperature_cpu=np.ones((4,), dtype=np.float32),
        top_p_cpu=np.ones((4,), dtype=np.float32),
        top_k_cpu=np.zeros((4,), dtype=np.int32),
        min_p_cpu=np.zeros((4,), dtype=np.float32),
        frequency_penalties_cpu=np.zeros((4,), dtype=np.float32),
        presence_penalties_cpu=np.zeros((4,), dtype=np.float32),
        repetition_penalties_cpu=np.ones((4,), dtype=np.float32),
    )

    assert sample_count == 3
    assert sampler_padded_num_reqs == 4
    assert total_tokens == 3
    np.testing.assert_array_equal(manager._sampler_gather_positions_cpu, np.array([0, 1, 2, 0], dtype=np.int32))
    np.testing.assert_array_equal(manager._sampler_active_mask_cpu, np.array([True, True, True, False]))
    np.testing.assert_array_equal(manager._sampler_seq_lens_cpu, np.array([5, 6, 7, 0], dtype=np.int32))


def test_runtime_bucket_validation_rejects_missing_spmd_bucket():
    manager = ExecutionManager.__new__(ExecutionManager)
    manager.model = _Model()

    class _ModelExecutor:
        @staticmethod
        def has_model_step(num_tokens: int, padded_num_reqs: int) -> bool:
            return False

    class _SamplerExecutor:
        @staticmethod
        def cache_key(*, padded_num_reqs: int) -> tuple[int, int, str, str]:
            return (0, int(padded_num_reqs), "sampler", "jit")

        @staticmethod
        def has(key: tuple[int, int, str, str]) -> bool:
            return True

    manager._model_executor = _ModelExecutor()
    manager._sampler_executor = _SamplerExecutor()

    with pytest.raises(RuntimeError, match="Missing precompiled eSurge bucket"):
        manager._require_precompiled_variants(
            num_tokens=16,
            padded_num_reqs=8,
            sampler_padded_num_reqs=4,
        )


def test_runtime_bucket_validation_accepts_precompiled_mpmd_split_bucket():
    manager = ExecutionManager.__new__(ExecutionManager)
    manager.model = _MpmdModel()

    class _ModelExecutor:
        supports_pipeline_model_step = False

        @staticmethod
        def has_model_step(num_tokens: int, padded_num_reqs: int) -> bool:
            return False

        @staticmethod
        def has_backbone(num_tokens: int, *, use_pipeline_runtime: bool = True) -> bool:
            return int(num_tokens) == 16 and bool(use_pipeline_runtime)

        @staticmethod
        def has_lm_head(padded_num_reqs: int) -> bool:
            return int(padded_num_reqs) == 8

    class _SamplerExecutor:
        @staticmethod
        def cache_key(*, padded_num_reqs: int) -> tuple[int, int, str, str]:
            return (0, int(padded_num_reqs), "sampler", "jit")

        @staticmethod
        def has(key: tuple[int, int, str, str]) -> bool:
            return key == (0, 4, "sampler", "jit")

    manager._model_executor = _ModelExecutor()
    manager._sampler_executor = _SamplerExecutor()

    manager._require_precompiled_variants(
        num_tokens=16,
        padded_num_reqs=8,
        sampler_padded_num_reqs=4,
        use_pipeline_runtime=True,
    )
