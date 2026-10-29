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
"""Execution-mode gradient accumulation vs the in-program scan, through the real distillation step. Runs on TPU."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.sharding import replicated_named_sharding
from easydel.trainers.distillation_trainer import _fn as distill_fn
from easydel.trainers.distillation_trainer.loss_groups import (
    LOSS_TOKEN_WEIGHTS_KEY,
    loss_group_token_weights,
    loss_supervision_mask,
)
from easydel.trainers.execution_accumulation import ExecutionAccumulation, build_execution_accumulation
from easydel.trainers.training_configurations import TrainingArguments
from jax import numpy as jnp

VOCAB, SEQ, BATCH, K, STEPS = 256, 64, 8, 4, 3


def make_states(seed: int = 0):
    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM

    def _model(s):
        config = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=VOCAB,
            max_position_embeddings=SEQ,
            sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        )
        config.attn_dtype = jnp.float32
        return LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(s),
        )

    teacher = _model(seed + 1).shard_model().to_state()
    student = _model(seed + 2).shard_model().to_state(trainable_selector="parameters").init_tx(optax.sgd(0.1))
    return student, teacher


def make_batch(with_token_weights: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    input_ids = rng.integers(1, VOCAB, size=(BATCH, SEQ)).astype(np.int32)
    completion = np.zeros((BATCH, SEQ), np.int32)
    for row in range(BATCH):
        completion[row, SEQ - 4 - 6 * row :] = 1
    batch = {
        "input_ids": input_ids,
        "attention_mask": np.ones((BATCH, SEQ), np.int32),
        "completion_mask": completion,
        "labels": input_ids.copy(),
    }
    if with_token_weights:
        group_ids = (np.arange(BATCH)[:, None] % 2).repeat(SEQ, axis=1).astype(np.int32)
        weights, _ = loss_group_token_weights(group_ids, loss_supervision_mask(batch, shift_tokens=True), [0.8, 0.2])
        batch[LOSS_TOKEN_WEIGHTS_KEY] = weights
    return batch


def step_fn(state, batch, teacher, gas, *, accumulate_into=None, apply_accumulated=None):
    return distill_fn.distillation_step(
        student_state=state,
        batch=batch,
        teacher_state=teacher,
        gradient_accumulation_steps=gas,
        is_training=True,
        temperature=2.0,
        alpha=0.8,
        logits_chunk_size=16,
        accumulate_into=accumulate_into,
        apply_accumulated=apply_accumulated,
    )


def build_programs(student, teacher):
    empty = replicated_named_sharding(student.model.mesh)
    in_shardings = (student.shardings, empty, teacher.shardings)
    scan_fn = spx.jit(
        step_fn,
        in_shardings=in_shardings,
        out_shardings=(student.shardings, empty),
        donate_argnums=(0,),
        static_argnums=(3,),
    )
    execution = ExecutionAccumulation(
        step_fn=step_fn,
        num_steps=K,
        in_shardings=in_shardings,
        state_shardings=student.shardings,
        empty_sharding=empty,
        static_argnums=(3,),
    )
    return scan_fn, execution


def compiled_temp_bytes(student, teacher, batch) -> dict[str, int]:
    """``temp_size_in_bytes`` of the scan GA program and of the execution grad / apply programs at ``K``."""
    scan_fn, execution = build_programs(student, teacher)
    with student.model.mesh:
        scan = scan_fn.lower(student, batch, teacher, K).compile().memory_analysis()
        microbatch = execution.microbatches(batch)[0]
        accumulator = execution.zeros(student)
        grad = execution.grad_fn.lower(student, microbatch, teacher, accumulator, K).compile().memory_analysis()
        _, metrics, _ = execution.grad_fn(student, microbatch, teacher, execution.zeros(student), K)
        apply = execution.apply_fn.lower(student, None, teacher, (accumulator, metrics), K).compile().memory_analysis()
    return {
        "scan": int(scan.temp_size_in_bytes),
        "grad": int(grad.temp_size_in_bytes),
        "apply": int(apply.temp_size_in_bytes),
    }


def _train(mode: str, batch):
    student, teacher = make_states()
    scan_fn, execution = build_programs(student, teacher)
    losses = []
    with student.model.mesh:
        for _ in range(STEPS):
            if mode == "scan":
                student, metrics = scan_fn(student, batch, teacher, K)
            else:
                student, metrics = execution.run(student, batch, (teacher,), (K,))
            losses.append(float(metrics.loss))
    return student, losses


@pytest.mark.parametrize("with_token_weights", [False, True])
def test_execution_mode_matches_scan_mode_after_n_steps(with_token_weights):
    batch = make_batch(with_token_weights)
    scan_state, scan_losses = _train("scan", batch)
    exec_state, exec_losses = _train("execution", batch)
    assert int(exec_state.step) == int(scan_state.step) == STEPS
    np.testing.assert_allclose(exec_losses, scan_losses, rtol=1e-5)
    start_state, _ = make_states()
    moved = 0.0
    for a, b, p0 in zip(
        jax.tree_util.tree_leaves(exec_state.graphstate),
        jax.tree_util.tree_leaves(scan_state.graphstate),
        jax.tree_util.tree_leaves(start_state.graphstate),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)
        moved = max(moved, float(np.abs(np.asarray(b) - np.asarray(p0)).max()))
    assert moved > 1e-4


def test_token_weighted_scan_differs_from_uniform_average():
    """The batch's microbatches carry unequal supervised counts, so the token weighting is load-bearing."""
    batch = make_batch()
    supervised = loss_supervision_mask(batch, shift_tokens=True)
    per_micro = supervised.reshape(K, BATCH // K, SEQ).sum(axis=(1, 2))
    assert len(set(per_micro.tolist())) == K


def test_grad_program_temporaries_below_scan_program():
    student, teacher = make_states()
    temps = compiled_temp_bytes(student, teacher, make_batch())
    assert temps["grad"] < temps["scan"], temps


def test_microbatch_split_refuses_indivisible_batch():
    student, teacher = make_states()
    _, execution = build_programs(student, teacher)
    batch = {k: v[:6] for k, v in make_batch().items()}
    with pytest.raises(ValueError, match="not divisible"):
        execution.microbatches(batch)


def test_build_returns_none_or_refuses():
    def trainer(**arguments):
        base = dict(gradient_accumulation_mode="execution", gradient_accumulation_steps=4, mpmd_scheduler=None)
        base.update(arguments)
        return SimpleNamespace(arguments=SimpleNamespace(**base), _has_buckets=False, state_shardings=None)

    kw = dict(step_fn=step_fn, in_shardings=(None, None), static_argnums=(), empty_sharding=None)
    assert build_execution_accumulation(trainer(gradient_accumulation_mode="scan"), **kw) is None
    assert build_execution_accumulation(trainer(gradient_accumulation_steps=1), **kw) is None
    with pytest.raises(ValueError, match="mpmd_scheduler"):
        build_execution_accumulation(trainer(mpmd_scheduler=object()), **kw)
    bucketed = trainer()
    bucketed._has_buckets = True
    with pytest.raises(ValueError, match="buckets"):
        build_execution_accumulation(bucketed, **kw)


def test_training_arguments_validate_the_mode():
    assert TrainingArguments().gradient_accumulation_mode == "scan"
    args = TrainingArguments(gradient_accumulation_mode="execution", gradient_accumulation_steps=2)
    assert TrainingArguments.from_dict(args.to_dict()).gradient_accumulation_mode == "execution"
    with pytest.raises(ValueError, match="gradient_accumulation_mode"):
        TrainingArguments(gradient_accumulation_mode="unrolled")


def test_jax_backend_is_an_accelerator():
    assert jax.default_backend() in ("tpu", "gpu")
