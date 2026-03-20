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

from easydel.inference.esurge.core.sampler import apply_history_penalties
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


def test_execution_manager_sample_tokens_forwards_penalty_state():
    device = jax.devices()[0]
    calls: dict[str, object] = {}

    def compiled(*args):
        calls["args"] = args
        return ("rng", "tokens", "mask")

    class _StubSamplerExecutor:
        def get_compiled(self, *, num_tokens: int, padded_num_reqs: int):
            calls["compile_key"] = (num_tokens, padded_num_reqs)
            return compiled

    manager = object.__new__(ExecutionManager)
    manager._sampler_executor = _StubSamplerExecutor()
    manager._empty_sharding = device
    manager._sampler_token_history_cpu = np.zeros((4, 6), dtype=np.int32)
    manager._sampler_zero_token_history = jax.device_put(manager._sampler_token_history_cpu, device)
    manager._sampler_zero_penalties = jax.device_put(np.zeros((4,), dtype=np.float32), device)
    manager._sampler_identity_repetition_penalties = jax.device_put(np.ones((4,), dtype=np.float32), device)

    result = ExecutionManager.sample_tokens(
        manager,
        num_tokens=2,
        padded_num_reqs=2,
        batch_metadata=object(),
        req_num_tokens_full=jnp.zeros((4,), dtype=jnp.int32),
        active_mask_full=jnp.array([True, True, False, False]),
        logits=jnp.zeros((2, 8), dtype=jnp.float32),
        rng_key=jax.random.PRNGKey(0),
        token_ids_cpu=np.array(
            [
                [1, 2, 0, 0, 0, 0],
                [3, 3, 4, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        need_penalties=True,
    )

    assert result == ("rng", "tokens", "mask")
    assert calls["compile_key"] == (2, 2)

    args = calls["args"]
    np.testing.assert_array_equal(np.asarray(args[5])[:2], np.array([[1, 2, 0, 0, 0, 0], [3, 3, 4, 0, 0, 0]]))


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
