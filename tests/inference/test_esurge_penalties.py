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

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from easydel.inference.esurge.core.sampler import (
    apply_history_penalties,
    apply_history_penalties_from_counts,
    build_history_token_counts,
    sample_tokens,
    update_token_counts,
)
from easydel.inference.esurge.core.sampling_metadata import SamplingMetadata
from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from easydel.inference.esurge.runners.execution_types import BatchMetadata
from easydel.inference.oai_proxies import InferenceApiRouter
from easydel.inference.openai_api_modules import ChatCompletionRequest, CompletionRequest
from easydel.inference.sampling_params import SamplingParams


def test_apply_history_penalties_supports_presence_and_repetition():
    logits = jnp.array([[2.0, 4.0, -3.0, 1.0]], dtype=jnp.float32)
    token_history = jnp.array([[1, 2, 2, 0]], dtype=jnp.int32)
    seq_lens = jnp.array([3], dtype=jnp.int32)
    active_mask = jnp.array([True])

    adjusted = apply_history_penalties(
        logits,
        token_history=token_history,
        seq_lens=seq_lens,
        active_mask=active_mask,
        presence_penalties=jnp.array([0.5], dtype=jnp.float32),
        frequency_penalties=jnp.array([0.25], dtype=jnp.float32),
        repetition_penalties=jnp.array([2.0], dtype=jnp.float32),
    )

    np.testing.assert_allclose(
        np.asarray(adjusted),
        np.array([[2.0, 1.625, -8.0, 1.0]], dtype=np.float32),
    )


def test_count_based_penalties_match_history_path():
    logits = jnp.array(
        [
            [2.0, 4.0, -3.0, 1.0, 0.5],
            [1.0, -2.0, 3.0, 0.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    token_history = jnp.array(
        [
            [1, 2, 2, 0],
            [4, 4, 0, 0],
        ],
        dtype=jnp.int32,
    )
    seq_lens = jnp.array([3, 2], dtype=jnp.int32)
    active_mask = jnp.array([True, True])
    presence = jnp.array([0.5, 0.25], dtype=jnp.float32)
    frequency = jnp.array([0.25, 0.5], dtype=jnp.float32)
    repetition = jnp.array([2.0, 1.5], dtype=jnp.float32)

    expected = apply_history_penalties(
        logits,
        token_history=token_history,
        seq_lens=seq_lens,
        active_mask=active_mask,
        presence_penalties=presence,
        frequency_penalties=frequency,
        repetition_penalties=repetition,
    )
    token_counts = build_history_token_counts(
        token_history=token_history,
        seq_lens=seq_lens,
        active_mask=active_mask,
        vocab_size=logits.shape[1],
    )
    actual = apply_history_penalties_from_counts(
        logits,
        token_counts=token_counts,
        active_mask=active_mask,
        presence_penalties=presence,
        frequency_penalties=frequency,
        repetition_penalties=repetition,
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_update_token_counts_only_increments_valid_rows():
    token_counts = jnp.zeros((4, 8), dtype=jnp.uint32)
    updated = update_token_counts(
        token_counts,
        row_indices=jnp.array([2, 1, 3], dtype=jnp.int32),
        sampled_tokens=jnp.array([5, 7, 5], dtype=jnp.int32),
        valid_mask=jnp.array([True, False, True]),
    )

    expected = np.zeros((4, 8), dtype=np.uint32)
    expected[2, 5] = 1
    expected[3, 5] = 1
    np.testing.assert_array_equal(np.asarray(updated), expected)


def test_sample_tokens_preserves_rng_identity_with_explicit_sampling_seeds():
    logits = jnp.array(
        [
            [2.0, 0.0, -1.0, 1.0],
            [0.5, 1.5, -0.5, 0.0],
            [1.0, -2.0, 3.0, 0.5],
            [0.2, 0.1, 0.0, 2.2],
        ],
        dtype=jnp.float32,
    )
    rng = jax.random.PRNGKey(7)
    full_metadata = SamplingMetadata(
        temperatures=jnp.ones((4, 1), dtype=jnp.float32),
        top_ps=jnp.ones((4,), dtype=jnp.float32),
        top_ks=jnp.zeros((4,), dtype=jnp.int32),
        min_ps=jnp.zeros((4,), dtype=jnp.float32),
        sampling_seeds=None,
        is_all_greedy=False,
        need_min_p_sampling=False,
        do_penalties=False,
        linear_penalty=None,
    )
    compact_metadata = SamplingMetadata(
        temperatures=jnp.ones((2, 1), dtype=jnp.float32),
        top_ps=jnp.ones((2,), dtype=jnp.float32),
        top_ks=jnp.zeros((2,), dtype=jnp.int32),
        min_ps=jnp.zeros((2,), dtype=jnp.float32),
        sampling_seeds=jnp.array([0, 3], dtype=jnp.int32),
        is_all_greedy=False,
        need_min_p_sampling=False,
        do_penalties=False,
        linear_penalty=None,
    )

    full = sample_tokens(logits, full_metadata, rng)
    compact = sample_tokens(logits[jnp.array([0, 3], dtype=jnp.int32)], compact_metadata, rng)

    np.testing.assert_array_equal(np.asarray(compact), np.asarray(full)[[0, 3]])


def test_execution_manager_sample_tokens_forwards_incremental_penalty_state():
    device = jax.devices()[0]
    calls: dict[str, object] = {}

    def compiled(*args):
        calls["args"] = args
        return (
            jax.random.PRNGKey(1),
            jnp.array([5, 7], dtype=jnp.int32),
            jnp.array([True, True]),
            jnp.ones((4, 8), dtype=jnp.uint32),
        )

    class _StubSamplerExecutor:
        def get_compiled(self, *, num_tokens: int, padded_num_reqs: int):
            calls["compile_key"] = (num_tokens, padded_num_reqs)
            return compiled

    manager = object.__new__(ExecutionManager)
    manager._sampler_executor = _StubSamplerExecutor()
    manager._empty_sharding = device
    manager._scatter_sampler_outputs = lambda tokens, valid, scatter_positions, padded_num_reqs: (tokens, valid)
    manager._sampler_zero_token_counts = jax.device_put(np.zeros((4, 8), dtype=np.uint32), device)
    manager._sampler_token_counts = jax.device_put(
        np.array(
            [
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 2, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint32,
        ),
        device,
    )
    manager._sampler_penalty_state_ready = True
    manager._ensure_sampler_penalty_state = lambda: calls.setdefault("rebuilt", True)

    result = ExecutionManager.sample_tokens(
        manager,
        num_tokens=2,
        padded_num_reqs=2,
        sampler_padded_num_reqs=2,
        sampler_num_reqs=2,
        sampler_total_tokens=2,
        req_num_tokens_full=jnp.array([3, 4, 0, 0], dtype=jnp.int32),
        logits=jnp.zeros((2, 8), dtype=jnp.float32),
        rng_key=jax.random.PRNGKey(0),
        gather_positions_cpu=np.array([0, 1, 0, 0], dtype=np.int32),
        sampling_seeds_cpu=np.array([0, 1, 9, 10], dtype=np.int32),
        scatter_positions_cpu=np.array([0, 1, 2, 3], dtype=np.int32),
        compact_window_row_indices_cpu=np.array([0, 1, 0, 0], dtype=np.int32),
        compact_scheduled_cpu=np.array([1, 1, 0, 0], dtype=np.int32),
        compact_seq_lens_cpu=np.array([3, 4, 0, 0], dtype=np.int32),
        compact_active_mask_cpu=np.array([True, True, False, False]),
        compact_temperature_cpu=np.array([0.7, 0.8, 1.0, 1.0], dtype=np.float32),
        compact_top_p_cpu=np.array([0.9, 0.95, 1.0, 1.0], dtype=np.float32),
        compact_top_k_cpu=np.array([32, 16, 0, 0], dtype=np.int32),
        compact_min_p_cpu=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        compact_frequency_penalties_cpu=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        compact_presence_penalties_cpu=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        compact_repetition_penalties_cpu=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        need_penalties=True,
    )

    assert calls["compile_key"] == (2, 2)
    assert calls["rebuilt"] is True

    args = calls["args"]
    np.testing.assert_array_equal(
        np.asarray(args[16])[:2],
        np.array(
            [
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 2, 1, 0, 0, 0, 0],
            ],
            dtype=np.uint32,
        ),
    )
    np.testing.assert_array_equal(np.asarray(args[7])[:2], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(args[17])[:2], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result[1]), np.array([5, 7], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result[2]), np.array([True, True]))


def test_batch_metadata_exposes_penalty_rows():
    packed_f32 = jnp.array(
        [
            [0.7, 0.0],
            [0.9, 1.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [1.4, 1.0],
        ],
        dtype=jnp.float32,
    )
    metadata = BatchMetadata(
        packed_qsl_seqlens=jnp.zeros((2, 3), dtype=jnp.int32),
        packed_i32_padded=jnp.zeros((3, 2), dtype=jnp.int32),
        packed_f32_padded=packed_f32,
        packed_misc_i32=jnp.array([2, 2, 0, 0, 0], dtype=jnp.int32),
        pages_tables=jnp.zeros((1, 1), dtype=jnp.int32),
        input_ids_buf=jnp.zeros((1,), dtype=jnp.int32),
        position_ids_buf=jnp.zeros((1,), dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(metadata.frequency_penalties), np.array([0.2, 0.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(metadata.presence_penalties), np.array([0.3, 0.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(metadata.repetition_penalties), np.array([1.4, 1.0], dtype=np.float32))


def test_oai_proxy_forwards_repetition_penalty():
    router = object.__new__(InferenceApiRouter)

    completion_params = router.build_oai_params_from_request(
        CompletionRequest(
            model="test-model",
            prompt="hello",
            repetition_penalty=1.7,
        )
    )
    assert completion_params["repetition_penalty"] == pytest.approx(1.7)

    chat_params = router.build_oai_params_from_chat_request(
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            repetition_penalty=1.3,
        )
    )
    assert chat_params["repetition_penalty"] == pytest.approx(1.3)


def test_sampling_params_rejects_non_positive_repetition_penalty():
    with pytest.raises(ValueError, match="repetition_penalty must be > 0"):
        SamplingParams(repetition_penalty=0.0)
