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

import functools
import types

import jax
import jax.numpy as jnp
import pytest

import easydel.infra.loss_utils as loss_utils_module

from easydel.infra.loss_utils import (
    ForCausalLMLoss,
    ForSequenceClassificationLoss,
    LossForwardPlan,
    LossConfig,
    LossMetrics,
    causal_lm_loss_chunked_lm_head,
    cross_entropy_blockwise_logits,
    fixed_cross_entropy,
    resolve_causal_lm_chunk_token_size,
    resolve_loss_strategy,
)


class _DummyCausalLMModule:
    class _Config:
        vocab_size = 32000
        partition_axis = None

    config = _Config()

    @staticmethod
    def compute_lm_logits(hidden_states):
        return hidden_states

    def __call__(self, input_ids=None, apply_lm_head=True):
        del input_ids, apply_lm_head
        return None


class _UnsupportedChunkedCausalLMModule:
    class _Config:
        vocab_size = 32000
        partition_axis = None

    config = _Config()

    @staticmethod
    def compute_lm_logits(hidden_states):
        return hidden_states

    def __call__(self, input_ids=None):
        return input_ids


def test_cross_entropy_blockwise_logits_respects_bf16_compute_dtype():
    logits = jnp.arange(2 * 3 * 8, dtype=jnp.bfloat16).reshape(2, 3, 8) / 10
    targets = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

    def loss_fn(x):
        return cross_entropy_blockwise_logits(
            x,
            targets,
            block_size=4,
            dtype=jnp.bfloat16,
        )[0]

    total_loss, total_z_loss, weight_sum, accuracy = cross_entropy_blockwise_logits(
        logits,
        targets,
        block_size=4,
        dtype=jnp.bfloat16,
    )
    gradients = jax.grad(loss_fn)(logits)

    assert total_loss.dtype == jnp.bfloat16
    assert total_z_loss.dtype == jnp.bfloat16
    assert weight_sum.dtype == jnp.bfloat16
    assert accuracy.dtype == jnp.bfloat16
    assert gradients.dtype == jnp.bfloat16


def test_cross_entropy_blockwise_logits_keeps_block_loop_checkpointed():
    logits = jnp.arange(2 * 3 * 8, dtype=jnp.bfloat16).reshape(2, 3, 8) / 10
    targets = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

    def loss_fn(x):
        return cross_entropy_blockwise_logits(
            x,
            targets,
            block_size=4,
            dtype=jnp.bfloat16,
        )[0]

    grad_jaxpr = str(jax.make_jaxpr(jax.grad(loss_fn))(logits))

    assert "remat2[" in grad_jaxpr or "checkpoint[" in grad_jaxpr


def test_causal_lm_loss_chunked_lm_head_matches_full_logits():
    hidden_states = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4) / 17
    kernel = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8) / 19
    labels = jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=jnp.int32)
    config = LossConfig(chunk_block_size=4)

    def lm_head_fn(x):
        return jnp.einsum("bth,hv->btv", x, kernel)

    expected = fixed_cross_entropy(
        source=lm_head_fn(hidden_states[:, :-1, :]),
        target=labels[:, 1:],
        attention_mask=attention_mask[:, 1:],
        config=config,
    )
    actual = causal_lm_loss_chunked_lm_head(
        hidden_states=hidden_states,
        labels=labels,
        lm_head_fn=lm_head_fn,
        vocab_size=kernel.shape[1],
        attention_mask=attention_mask,
        config=config,
        token_chunk_size=2,
    )

    assert jnp.allclose(actual.loss, expected.loss, atol=1e-5)
    assert jnp.allclose(actual.z_loss, expected.z_loss, atol=1e-5)
    assert jnp.allclose(actual.weight_sum, expected.weight_sum, atol=1e-5)
    assert jnp.allclose(actual.accuracy, expected.accuracy, atol=1e-5)


def test_causal_lm_loss_chunked_lm_head_keeps_projection_checkpointed():
    hidden_states = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4) / 17
    kernel = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8) / 19
    labels = jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=jnp.int32)
    config = LossConfig(chunk_block_size=4)

    def loss_fn(x):
        return causal_lm_loss_chunked_lm_head(
            hidden_states=x,
            labels=labels,
            lm_head_fn=lambda h: jnp.einsum("bth,hv->btv", h, kernel),
            vocab_size=kernel.shape[1],
            config=config,
            token_chunk_size=2,
        ).loss

    grad_jaxpr = str(jax.make_jaxpr(jax.grad(loss_fn))(hidden_states))

    assert "remat2[" in grad_jaxpr or "checkpoint[" in grad_jaxpr


def test_causal_lm_loss_chunked_lm_head_supports_fp32_compute_dtype_with_bf16_hidden_states():
    hidden_states = (jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4) / 17).astype(jnp.bfloat16)
    kernel = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8) / 19
    labels = jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=jnp.int32)
    config = LossConfig(chunk_block_size=4, compute_dtype="fp32")

    metrics = causal_lm_loss_chunked_lm_head(
        hidden_states=hidden_states,
        labels=labels,
        lm_head_fn=lambda h: jnp.einsum("bth,hv->btv", h.astype(jnp.float32), kernel),
        vocab_size=kernel.shape[1],
        config=config,
        token_chunk_size=2,
    )

    assert metrics.loss.dtype == jnp.float32
    assert metrics.z_loss.dtype == jnp.float32
    assert metrics.weight_sum.dtype == jnp.float32
    assert metrics.accuracy.dtype == jnp.float32


def test_causal_lm_loss_chunked_lm_head_preserves_decoder_loss_weights():
    hidden_states = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4) / 17
    kernel = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8) / 19
    labels = jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=jnp.int32)
    decoder_loss_weights = jnp.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    config = LossConfig(chunk_block_size=4)

    def lm_head_fn(x):
        return jnp.einsum("bth,hv->btv", x, kernel)

    shifted_weights = decoder_loss_weights[:, 1:]
    expected = fixed_cross_entropy(
        source=lm_head_fn(hidden_states[:, :-1, :]),
        target=labels[:, 1:],
        config=config,
        batch={
            "decoder_target_tokens": labels[:, 1:],
            "decoder_loss_weights": shifted_weights,
        },
    )
    actual = causal_lm_loss_chunked_lm_head(
        hidden_states=hidden_states,
        labels=labels,
        lm_head_fn=lm_head_fn,
        vocab_size=kernel.shape[1],
        config=config,
        token_chunk_size=2,
        batch={"decoder_loss_weights": decoder_loss_weights},
    )

    assert jnp.allclose(actual.loss, expected.loss, atol=1e-5)
    assert jnp.allclose(actual.z_loss, expected.z_loss, atol=1e-5)
    assert jnp.allclose(actual.weight_sum, expected.weight_sum, atol=1e-5)
    assert jnp.allclose(actual.accuracy, expected.accuracy, atol=1e-5)


def test_causal_lm_loss_chunked_lm_head_disables_inner_ce_chunking(monkeypatch: pytest.MonkeyPatch):
    hidden_states = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4) / 17
    labels = jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=jnp.int32)
    seen_configs: list[tuple[int | None, int | None, int | None]] = []

    def fake_fixed_cross_entropy(*, source, target, config=None, **kwargs):
        del source, target, kwargs
        seen_configs.append((config.chunk_vocab_size, config.chunk_block_size, config.chunk_token_size))
        one = jnp.array(1.0, dtype=jnp.float32)
        return LossMetrics(loss=one, z_loss=one, weight_sum=one, accuracy=one)

    monkeypatch.setattr(loss_utils_module, "fixed_cross_entropy", fake_fixed_cross_entropy)

    causal_lm_loss_chunked_lm_head(
        hidden_states=hidden_states,
        labels=labels,
        lm_head_fn=lambda x: x,
        vocab_size=hidden_states.shape[-1],
        config=LossConfig(chunk_vocab_size=3, chunk_block_size=4, chunk_token_size=2),
    )

    assert seen_configs
    assert all(chunk_vocab_size is None for chunk_vocab_size, _, _ in seen_configs)
    assert all(chunk_block_size is None for _, chunk_block_size, _ in seen_configs)
    assert all(chunk_token_size is None for _, _, chunk_token_size in seen_configs)


def test_causal_lm_loss_strategy_disables_lm_head_for_large_blockwise_loss():
    strategy = resolve_loss_strategy(ForCausalLMLoss)
    labels = jnp.ones((2, 8193), dtype=jnp.int32)
    plan = strategy.plan_forward(
        module=_DummyCausalLMModule(),
        labels=labels,
        loss_config=LossConfig(chunk_block_size=4096),
        batch={},
        loss_kwargs={},
    )

    assert plan.forward_kwargs == {"apply_lm_head": False}


def test_causal_lm_loss_strategy_honors_explicit_token_chunk_size_loss_kwarg():
    strategy = resolve_loss_strategy(ForCausalLMLoss)
    labels = jnp.ones((2, 8), dtype=jnp.int32)
    plan = strategy.plan_forward(
        module=_DummyCausalLMModule(),
        labels=labels,
        loss_config=LossConfig(chunk_block_size=None, chunk_vocab_size=None, chunk_token_size=None),
        batch={},
        loss_kwargs={"token_chunk_size": 2},
    )

    assert plan.forward_kwargs == {"apply_lm_head": False}


def test_causal_lm_loss_strategy_uses_module_compute_lm_logits_postprocessing():
    kernel = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8) / 19

    class _Module:
        class _Config:
            vocab_size = 8
            partition_axis = None

        config = _Config()

        @staticmethod
        def compute_lm_logits(hidden_states):
            logits = jnp.einsum("bth,hv->btv", hidden_states, kernel)
            return logits * 0.5

    hidden_states = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4) / 17
    labels = jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=jnp.int32)
    strategy = resolve_loss_strategy(ForCausalLMLoss)
    expected = fixed_cross_entropy(
        source=_Module.compute_lm_logits(hidden_states[:, :-1, :]),
        target=labels[:, 1:],
        config=LossConfig(chunk_block_size=4),
    )

    actual = strategy.compute(
        module=_Module(),
        outputs=types.SimpleNamespace(last_hidden_state=hidden_states),
        labels=labels,
        loss_config=LossConfig(chunk_block_size=4),
        batch={},
        loss_kwargs={"token_chunk_size": 2},
        paxis=None,
        forward_plan=LossForwardPlan(forward_kwargs={"apply_lm_head": False}),
    )

    assert jnp.allclose(actual.loss, expected.loss, atol=1e-5)
    assert jnp.allclose(actual.z_loss, expected.z_loss, atol=1e-5)
    assert jnp.allclose(actual.weight_sum, expected.weight_sum, atol=1e-5)
    assert jnp.allclose(actual.accuracy, expected.accuracy, atol=1e-5)


def test_causal_lm_loss_strategy_uses_module_prepare_lm_head_inputs():
    class _Module:
        class _Config:
            vocab_size = 8
            partition_axis = None

        config = _Config()

        @staticmethod
        def prepare_lm_head_inputs(hidden_states):
            return hidden_states * 0.5

        @staticmethod
        def compute_lm_logits(hidden_states):
            return hidden_states

    hidden_states = jnp.arange(2 * 5 * 8, dtype=jnp.float32).reshape(2, 5, 8) / 17
    labels = jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=jnp.int32)
    strategy = resolve_loss_strategy(ForCausalLMLoss)

    expected = fixed_cross_entropy(
        source=_Module.prepare_lm_head_inputs(hidden_states[:, :-1, :]),
        target=labels[:, 1:],
        config=LossConfig(chunk_block_size=4),
    )
    actual = strategy.compute(
        module=_Module(),
        outputs=types.SimpleNamespace(last_hidden_state=hidden_states),
        labels=labels,
        loss_config=LossConfig(chunk_block_size=4),
        batch={},
        loss_kwargs={"token_chunk_size": 2},
        paxis=None,
        forward_plan=LossForwardPlan(forward_kwargs={"apply_lm_head": False}),
    )

    assert jnp.allclose(actual.loss, expected.loss, atol=1e-5)
    assert jnp.allclose(actual.z_loss, expected.z_loss, atol=1e-5)
    assert jnp.allclose(actual.weight_sum, expected.weight_sum, atol=1e-5)
    assert jnp.allclose(actual.accuracy, expected.accuracy, atol=1e-5)


def test_causal_lm_loss_strategy_forwards_num_items_in_batch_from_batch():
    kernel = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8) / 19

    class _Module:
        class _Config:
            vocab_size = 8
            partition_axis = None

        config = _Config()

        @staticmethod
        def compute_lm_logits(hidden_states):
            return jnp.einsum("bth,hv->btv", hidden_states, kernel)

    hidden_states = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4) / 17
    labels = jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=jnp.int32)
    batch = {"num_items_in_batch": 3}
    strategy = resolve_loss_strategy(ForCausalLMLoss)

    expected = causal_lm_loss_chunked_lm_head(
        hidden_states=hidden_states,
        labels=labels,
        lm_head_fn=_Module.compute_lm_logits,
        vocab_size=8,
        config=LossConfig(chunk_block_size=4),
        token_chunk_size=2,
        num_items_in_batch=3,
    )
    actual = strategy.compute(
        module=_Module(),
        outputs=types.SimpleNamespace(last_hidden_state=hidden_states),
        labels=labels,
        loss_config=LossConfig(chunk_block_size=4),
        batch=batch,
        loss_kwargs={"token_chunk_size": 2},
        paxis=None,
        forward_plan=LossForwardPlan(forward_kwargs={"apply_lm_head": False}),
    )

    assert jnp.allclose(actual.loss, expected.loss, atol=1e-5)
    assert jnp.allclose(actual.z_loss, expected.z_loss, atol=1e-5)
    assert jnp.allclose(actual.weight_sum, expected.weight_sum, atol=1e-5)
    assert jnp.allclose(actual.accuracy, expected.accuracy, atol=1e-5)


def test_legacy_loss_strategy_keeps_default_forward_plan():
    strategy = resolve_loss_strategy(ForSequenceClassificationLoss)
    plan = strategy.plan_forward(
        module=object(),
        labels=jnp.array([0, 1], dtype=jnp.int32),
        loss_config=LossConfig(num_labels=2),
        batch={},
        loss_kwargs={},
    )

    assert plan.forward_kwargs == {}


def test_causal_lm_loss_strategy_skips_modules_without_apply_lm_head_forward_flag():
    strategy = resolve_loss_strategy(ForCausalLMLoss)
    labels = jnp.ones((2, 8193), dtype=jnp.int32)
    plan = strategy.plan_forward(
        module=_UnsupportedChunkedCausalLMModule(),
        labels=labels,
        loss_config=LossConfig(chunk_block_size=4096),
        batch={},
        loss_kwargs={},
    )

    assert plan.forward_kwargs == {}


def test_resolve_loss_strategy_unwraps_wrapped_causal_lm_loss():
    @functools.wraps(ForCausalLMLoss)
    def wrapped_loss(*args, **kwargs):
        return ForCausalLMLoss(*args, **kwargs)

    strategy = resolve_loss_strategy(wrapped_loss)

    assert type(strategy).__name__ == "CausalLMLossStrategy"


def _make_label_smoothing_case():
    logits = jnp.array(
        [
            [
                [0.1, 0.4, -0.7, 0.6, 0.3, -0.2, 0.5],
                [0.2, -0.1, 0.8, 0.7, -0.5, 0.6, 0.1],
                [0.9, 0.1, -0.4, 0.3, 0.5, -0.6, 0.7],
                [-0.3, 0.2, 0.4, -0.1, 0.8, 0.5, -0.2],
            ],
            [
                [0.6, -0.2, 0.3, 0.9, -0.4, 0.1, 0.7],
                [0.5, 0.8, -0.3, 0.2, 0.4, -0.6, 0.1],
                [-0.2, 0.7, 0.6, -0.5, 0.3, 0.9, 0.4],
                [0.4, -0.7, 0.2, 0.5, 0.6, 0.3, -0.1],
            ],
        ],
        dtype=jnp.float32,
    )
    targets = jnp.array(
        [
            [1, 3, 4, -100],
            [0, 2, 5, 4],
        ],
        dtype=jnp.int32,
    )
    weights = jnp.array(
        [
            [1.0, 0.5, 1.0, 0.0],
            [0.3, 1.0, 0.8, 0.2],
        ],
        dtype=jnp.float32,
    )
    batch = {
        "decoder_target_tokens": targets,
        "decoder_loss_weights": weights,
    }
    dense_config = LossConfig(label_smoothing=0.1, z_loss=1e-4, chunk_block_size=None)
    return logits, targets, batch, dense_config


def _assert_metrics_close(actual, expected):
    assert jnp.allclose(actual.loss, expected.loss, atol=1e-5)
    assert jnp.allclose(actual.z_loss, expected.z_loss, atol=1e-5)
    assert jnp.allclose(actual.weight_sum, expected.weight_sum, atol=1e-5)
    assert jnp.allclose(actual.accuracy, expected.accuracy, atol=1e-5)


def test_fixed_cross_entropy_default_blockwise_matches_dense_label_smoothing():
    logits, targets, batch, dense_config = _make_label_smoothing_case()

    expected = fixed_cross_entropy(
        source=logits,
        target=targets,
        config=dense_config,
        batch=batch,
    )
    actual = fixed_cross_entropy(
        source=logits,
        target=targets,
        config=LossConfig(label_smoothing=0.1, z_loss=1e-4),
        batch=batch,
    )

    _assert_metrics_close(actual, expected)


def test_fixed_cross_entropy_chunk_vocab_matches_dense_label_smoothing():
    logits, targets, batch, dense_config = _make_label_smoothing_case()

    expected = fixed_cross_entropy(
        source=logits,
        target=targets,
        config=dense_config,
        batch=batch,
    )
    actual = fixed_cross_entropy(
        source=logits,
        target=targets,
        config=LossConfig(label_smoothing=0.1, z_loss=1e-4, chunk_block_size=None, chunk_vocab_size=3),
        batch=batch,
    )

    _assert_metrics_close(actual, expected)


def test_fixed_cross_entropy_chunk_tokens_matches_dense_label_smoothing():
    logits, targets, batch, dense_config = _make_label_smoothing_case()

    expected = fixed_cross_entropy(
        source=logits,
        target=targets,
        config=dense_config,
        batch=batch,
    )
    actual = fixed_cross_entropy(
        source=logits,
        target=targets,
        config=LossConfig(label_smoothing=0.1, z_loss=1e-4, chunk_block_size=None, chunk_token_size=3),
        batch=batch,
    )

    _assert_metrics_close(actual, expected)


def test_resolve_causal_lm_chunk_token_size_respects_budget():
    hidden_states = jax.ShapeDtypeStruct((128, 2048, 4096), jnp.bfloat16)

    chunk = resolve_causal_lm_chunk_token_size(hidden_states, vocab_size=256_000)
    fp32_chunk = resolve_causal_lm_chunk_token_size(
        hidden_states,
        vocab_size=256_000,
        config=LossConfig(compute_dtype="fp32"),
    )

    assert chunk == 256
    assert fp32_chunk == 128
