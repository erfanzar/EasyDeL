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

"""Regression tests for PR #306 RL / math / GA / OPD / GRPO / PPO fixes."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import easydel as ed
import easydel.caching.shared_prefix as shared_prefix
import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers._shared.esurge_generation import outputs_to_arrays
from easydel.trainers._shared.off_policy import check_startup_behavior_parity
from easydel.trainers.base_trainer import GenerationResults
from easydel.trainers.distillation_trainer._fn import _per_token_kl_and_entropy, chunked_distillation_loss
from easydel.trainers.group_relative_policy_optimization import _fn as grpo_fn
from easydel.trainers.group_relative_policy_optimization._fn import (
    _compute_grpo_policy_loss_terms,
    get_per_token_logps,
    grpo_step,
    validate_prefix_shared_model,
)
from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer, _ScoredRollout
from easydel.trainers.grpo_replay_buffer_trainer.grpo_replay_buffer_trainer import GRPOWithReplayBufferTrainer
from easydel.trainers.grpo_replay_buffer_trainer.replay_buffer import _ReplayBuffer
from easydel.trainers.on_policy_distillation_trainer import OnPolicyDistillationConfig
from easydel.trainers.proximal_policy_optimization_trainer._fn import _token_normalizer, ppo_step
from easydel.trainers.training_utils import (
    SCHEDULED_UPDATE_GATE_KEY,
    _run_scheduled_value_and_grad,
    compile_trainer_step,
    normalize_generation_model_kwargs,
)
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.test_util import check_grads

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_llama(seed: int = 0):
    cfg = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        mlp_bias=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
    )
    model = ed.LlamaForCausalLM(config=cfg, rngs=spx.Rngs(seed), dtype=jnp.float32, param_dtype=jnp.float32)
    return cfg, model


def _grpo_batch(prompts: int, num_generations: int, prompt_len: int = 5, completion_len: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    prompt_mask = np.ones((prompts, prompt_len), dtype=np.int32)
    prompt_mask[0, :2] = 0
    prompt_ids = np.where(prompt_mask > 0, rng.integers(1, 60, (prompts, prompt_len)), 0).astype(np.int32)
    rows = prompts * num_generations
    lengths = rng.integers(1, completion_len + 1, rows)
    completion_mask = (np.arange(completion_len)[None, :] < lengths[:, None]).astype(np.float32)
    completion_ids = np.where(completion_mask > 0, rng.integers(1, 60, (rows, completion_len)), 0).astype(np.int32)
    advantages = rng.standard_normal(rows).astype(np.float32)
    return {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.asarray(prompt_mask),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(completion_mask),
        "advantages": jnp.asarray(advantages),
        "num_items_in_batch": jnp.asarray(completion_mask.sum(), dtype=jnp.float32),
    }


# ---------------------------------------------------------------------------
# #7: Prefix-shared GRPO under gradient_accumulation_steps > 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompts,num_generations,ga_steps", [(4, 2, 2), (4, 4, 2)])
def test_pr306_7_prefix_shared_grpo_lockstep_ga_matches_repeated_prompt_and_keeps_sharing(
    monkeypatch, prompts, num_generations, ga_steps
):
    """Prefix-shared GRPO with GA > 1 slices prompt groups in lockstep without disabling sharing."""
    cfg, model = _tiny_llama(seed=11)
    batch = _grpo_batch(prompts=prompts, num_generations=num_generations, seed=17)

    # Confirm the dense grouped-fork path never materializes per-completion prefix copies
    # and that get_per_token_logps_prefix_shared receives the real num_generations > 1.
    def forbid_copy(self, **kwargs):
        raise AssertionError("prefix copy must not run on vanilla grouped-fork path")

    observed_generations: list[int] = []
    real_prefix_shared = grpo_fn.get_per_token_logps_prefix_shared

    def spy_prefix_shared(mod, p_ids, p_mask, c_ids, c_mask, eff_g, *args, **kwargs):
        observed_generations.append(int(eff_g))
        assert int(p_ids.shape[0]) == prompts // ga_steps
        assert int(c_ids.shape[0]) == (prompts * num_generations) // ga_steps
        return real_prefix_shared(mod, p_ids, p_mask, c_ids, c_mask, eff_g, *args, **kwargs)

    monkeypatch.setattr(shared_prefix.PrefixForkView, "concatenate_to_cache", forbid_copy)
    monkeypatch.setattr(grpo_fn, "get_per_token_logps_prefix_shared", spy_prefix_shared)

    with jax.default_matmul_precision("highest"), cfg.mesh:
        state = EasyDeLState.create(model=model, tx=optax.sgd(1.0), init_opt_state=True)
        before = [np.asarray(x) for x in jax.tree_util.tree_leaves(state.graphstate)]

        ref_ga1_state, ref_ga1_metrics = grpo_step(
            state,
            batch,
            num_generations,
            0.0,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=1,
            is_training=True,
            loss_type="dapo",
            prefix_shared_forward=False,
        )
        ref_ga_state, ref_ga_metrics = grpo_step(
            state,
            batch,
            num_generations,
            0.0,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=ga_steps,
            is_training=True,
            loss_type="dapo",
            prefix_shared_forward=False,
        )
        shared_ga_state, shared_ga_metrics = grpo_step(
            state,
            batch,
            num_generations,
            0.0,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=ga_steps,
            is_training=True,
            loss_type="dapo",
            prefix_shared_forward=True,
        )

    assert observed_generations and all(g == num_generations for g in observed_generations)
    np.testing.assert_allclose(float(shared_ga_metrics.loss), float(ref_ga_metrics.loss), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(shared_ga_metrics.loss), float(ref_ga1_metrics.loss), rtol=1e-4, atol=1e-6)
    for b, ref1, refk, sharedk in zip(
        before,
        [np.asarray(x) for x in jax.tree_util.tree_leaves(ref_ga1_state.graphstate)],
        [np.asarray(x) for x in jax.tree_util.tree_leaves(ref_ga_state.graphstate)],
        [np.asarray(x) for x in jax.tree_util.tree_leaves(shared_ga_state.graphstate)],
        strict=True,
    ):
        np.testing.assert_allclose(b - sharedk, b - refk, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(b - sharedk, b - ref1, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("prompts,num_generations,ga_steps", [(2, 4, 4), (3, 4, 2)])
def test_pr306_7_prefix_shared_grpo_rejects_incompatible_ga_group_split(prompts, num_generations, ga_steps):
    cfg, model = _tiny_llama(seed=13)
    batch = _grpo_batch(prompts=prompts, num_generations=num_generations, seed=19)
    with cfg.mesh:
        state = EasyDeLState.create(model=model, tx=optax.sgd(1.0), init_opt_state=True)
        with pytest.raises(ValueError, match="whole prompt groups"):
            grpo_step(
                state,
                batch,
                num_generations,
                0.0,
                partition_spec=PartitionSpec(),
                gradient_accumulation_steps=ga_steps,
                is_training=True,
                loss_type="dapo",
                prefix_shared_forward=True,
            )


def test_pr306_7_grpo_config_prefix_shared_ga_mode_validation():
    cfg_scan = GRPOConfig(prefix_shared_forward=True, gradient_accumulation_steps=2, gradient_accumulation_mode="scan")
    assert cfg_scan.prefix_shared_forward is True
    with pytest.raises(ValueError, match="execution"):
        GRPOConfig(prefix_shared_forward=True, gradient_accumulation_steps=2, gradient_accumulation_mode="execution")


# ---------------------------------------------------------------------------
# #9: MPMD scheduled GRPO with all-discarded outer accumulation
# ---------------------------------------------------------------------------


def test_pr306_9_scheduled_all_discarded_outer_accumulation_preserves_has_behavior_and_zero_grads():
    """All-discarded outer accumulation keeps zero loss/grads while preserving grpo_gate_sums/*."""
    graphstate = {"w": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}
    batch = {
        "x": jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
        "difficulty_weights": jnp.zeros((4,), dtype=jnp.float32),
        "behavior_per_token_logps": jnp.zeros((4, 2), dtype=jnp.float32),
        "behavior_parity_gate": jnp.asarray(1, dtype=jnp.int32),
    }

    def fake_vag(tree, mb):
        # Simulate an empty/all-discarded chunk returning non-zero loss/grads and even a NaN non-gate aux,
        # alongside valid grpo_gate_sums/* indicating behavior logps were present.
        del tree, mb
        loss = jnp.asarray(3.5, dtype=jnp.float32)
        grads = {"w": jnp.asarray([9.0, -4.0], dtype=jnp.float32)}
        aux = {
            "z_loss": jnp.asarray(jnp.nan, dtype=jnp.float32),
            "grpo_gate_sums/has_behavior": jnp.asarray(1.0, dtype=jnp.float32),
            "grpo_gate_sums/eligible_rows": jnp.asarray(0.0, dtype=jnp.float32),
            "grpo_gate_sums/accepted_rows": jnp.asarray(0.0, dtype=jnp.float32),
            "grpo_gate_sums/tokens": jnp.asarray(0.0, dtype=jnp.float32),
            "grpo_gate_sums/k3_sum": jnp.asarray(0.0, dtype=jnp.float32),
            "grpo_gate_sums/tail_count": jnp.asarray(0.0, dtype=jnp.float32),
        }
        return loss, grads, aux

    loss, grads, aux = _run_scheduled_value_and_grad(
        value_and_grad=fake_vag,
        graphstate=graphstate,
        batch=batch,
        batch_size=4,
        minibatch_size=2,
        microbatch_weight_fn=lambda _t, mb: jnp.sum(mb["difficulty_weights"]),
    )

    assert float(loss) == 0.0
    np.testing.assert_array_equal(np.asarray(grads["w"]), [0.0, 0.0])
    # Non-gate NaN aux stays sanitized to 0.0 while gate-sum indicator is preserved.
    assert float(aux["z_loss"]) == 0.0
    assert float(aux["grpo_gate_sums/has_behavior"]) == pytest.approx(1.0)

    call = SimpleNamespace(
        get=lambda name, default=None: {
            "off_policy_min_acceptance": 0.5,
            "startup_behavior_k3_max": 0.1,
            "startup_behavior_tail_fraction_max": 0.01,
        }.get(name, default),
        batch=batch,
    )
    finalized = grpo_fn._grpo_scheduled_finalize_aux(call, aux)
    assert float(finalized["off_policy/behavior_k3"]) == pytest.approx(0.0)
    assert float(finalized["off_policy/tail_fraction"]) == pytest.approx(0.0)
    assert float(finalized["off_policy/accepted_fraction"]) == pytest.approx(1.0)
    assert float(finalized["off_policy/step_skipped"]) == 0.0
    assert float(finalized[SCHEDULED_UPDATE_GATE_KEY]) == 1.0
    check_startup_behavior_parity(finalized, k3_max=0.1, tail_fraction_max=0.01)


def test_pr306_9_scheduled_all_zero_weights_still_fails_closed_when_behavior_missing_and_catches_bad_k3():
    graphstate = {"w": jnp.asarray(1.0, dtype=jnp.float32)}
    batch_no_behavior = {"x": jnp.zeros((4, 2), dtype=jnp.float32), "w": jnp.zeros((4,), dtype=jnp.float32)}

    def vag_no_behavior(_t, _mb):
        return (
            jnp.asarray(1.0, jnp.float32),
            {"w": jnp.asarray(1.0, jnp.float32)},
            {grpo_fn._GRPO_GATE_AUX_PREFIX + k: jnp.asarray(0.0, jnp.float32) for k in grpo_fn._GRPO_GATE_SUM_KEYS},
        )

    _, _, aux_missing = _run_scheduled_value_and_grad(
        value_and_grad=vag_no_behavior,
        graphstate=graphstate,
        batch=batch_no_behavior,
        batch_size=4,
        minibatch_size=2,
        microbatch_weight_fn=lambda _t, mb: jnp.sum(mb["w"]),
    )
    call_missing = SimpleNamespace(
        get=lambda name, default=None: {"off_policy_min_acceptance": 0.5}.get(name, default),
        batch=batch_no_behavior,
    )
    finalized_missing = grpo_fn._grpo_scheduled_finalize_aux(call_missing, aux_missing)
    assert float(finalized_missing[SCHEDULED_UPDATE_GATE_KEY]) == 0.0
    assert float(finalized_missing["off_policy/step_skipped"]) == 1.0

    # Zero difficulty weights with real tokens carrying bad K3 must still trip the startup parity gate.
    batch_bad_k3 = {
        **batch_no_behavior,
        "behavior_per_token_logps": jnp.zeros((4, 2), dtype=jnp.float32),
        "behavior_parity_gate": jnp.asarray(1, dtype=jnp.int32),
    }

    def vag_bad_k3(_t, _mb):
        return (
            jnp.asarray(1.0, jnp.float32),
            {"w": jnp.asarray(1.0, jnp.float32)},
            {
                "grpo_gate_sums/has_behavior": jnp.asarray(1.0, jnp.float32),
                "grpo_gate_sums/eligible_rows": jnp.asarray(2.0, jnp.float32),
                "grpo_gate_sums/accepted_rows": jnp.asarray(2.0, jnp.float32),
                "grpo_gate_sums/tokens": jnp.asarray(4.0, jnp.float32),
                "grpo_gate_sums/k3_sum": jnp.asarray(2.0, jnp.float32),  # mean k3 = 0.5 > 0.1
                "grpo_gate_sums/tail_count": jnp.asarray(0.0, jnp.float32),
            },
        )

    _, _, aux_bad = _run_scheduled_value_and_grad(
        value_and_grad=vag_bad_k3,
        graphstate=graphstate,
        batch=batch_bad_k3,
        batch_size=4,
        minibatch_size=2,
        microbatch_weight_fn=lambda _t, mb: jnp.sum(mb["w"]),
    )
    call_bad = SimpleNamespace(
        get=lambda name, default=None: {
            "startup_behavior_k3_max": 0.1,
            "startup_behavior_tail_fraction_max": 0.01,
        }.get(name, default),
        batch=batch_bad_k3,
    )
    finalized_bad = grpo_fn._grpo_scheduled_finalize_aux(call_bad, aux_bad)
    assert float(finalized_bad["off_policy/behavior_k3"]) == pytest.approx(0.5)
    assert float(finalized_bad[SCHEDULED_UPDATE_GATE_KEY]) == 0.0


# ---------------------------------------------------------------------------
# #10: Async lookahead parity-gate refresh on learner thread
# ---------------------------------------------------------------------------


def test_pr306_10_refresh_behavior_parity_gate_clears_prefetched_lookahead_batches():
    """Prefetched async batches stamped with gate=1 before step-0 end are refreshed to gate=0."""
    cfg, model = _tiny_llama(seed=21)
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(
        esurge_importance_sampling_correction=True,
        startup_behavior_k3_max=0.05,
        startup_behavior_tail_fraction_max=0.01,
        sync_ref_model=False,
    )
    trainer.ref_state = None
    trainer._startup_behavior_parity_checked = False

    base_batch = _grpo_batch(prompts=2, num_generations=2, seed=23)
    # Large drift vs policy logps -> would fail startup_behavior_k3_max=0.05 if gate==1.
    base_batch["behavior_per_token_logps"] = jnp.full_like(base_batch["completion_mask"], -20.0)

    # Simulate background worker prefetching BOTH step-0 and step-1 before step-0 runs on_step_end.
    step0_batch, _ = trainer._with_behavior_parity_gate(dict(base_batch), {})
    step1_prefetched, _ = trainer._with_behavior_parity_gate(dict(base_batch), {})
    assert int(step0_batch["behavior_parity_gate"]) == 1
    assert int(step1_prefetched["behavior_parity_gate"]) == 1

    # Before step 0 executes, refresh keeps gate == 1 (step 0 still enforces startup parity).
    step0_ready = trainer._refresh_behavior_parity_gate_for_step(step0_batch)
    assert int(step0_ready["behavior_parity_gate"]) == 1

    with cfg.mesh:
        state = EasyDeLState.create(model=model, tx=optax.sgd(1.0), init_opt_state=True)
        step0_state, step0_metrics = grpo_step(
            state,
            step0_ready,
            2,
            0.0,
            partition_spec=PartitionSpec(),
            is_training=True,
            loss_type="dapo",
            startup_behavior_k3_max=0.05,
            startup_behavior_tail_fraction_max=0.01,
        )
        assert float(step0_metrics.other_metrics["off_policy/step_skipped"]) == 1.0
        assert int(step0_state.step) == int(state.step)

        # Simulate step 0 completing on_step_end with passing parity metrics.
        trainer.on_step_end(
            state,
            SimpleNamespace(other_metrics={"off_policy/behavior_k3": 0.0, "off_policy/tail_fraction": 0.0}),
            step=0,
        )
        assert trainer._startup_behavior_parity_checked is True

        # Now the consumer thread refreshes the already-prefetched step-1 batch right before execution.
        step1_ready = trainer._refresh_behavior_parity_gate_for_step(step1_prefetched)
        assert set(step1_ready) == set(step0_ready)
        assert int(step1_ready["behavior_parity_gate"]) == 0
        assert step1_ready["behavior_parity_gate"].dtype == jnp.int32
        assert step1_ready["behavior_parity_gate"].shape == ()

        step1_state, step1_metrics = grpo_step(
            state,
            step1_ready,
            2,
            0.0,
            partition_spec=PartitionSpec(),
            is_training=True,
            loss_type="dapo",
            startup_behavior_k3_max=0.05,
            startup_behavior_tail_fraction_max=0.01,
        )
        assert float(step1_metrics.other_metrics["off_policy/step_skipped"]) == 0.0
        assert int(step1_state.step) == int(state.step) + 1


# ---------------------------------------------------------------------------
# #23: outputs_to_arrays materializes completion_weight_versions on mesh
# ---------------------------------------------------------------------------


def test_pr306_23_outputs_to_arrays_materializes_weight_versions_on_mesh():
    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    comp = SimpleNamespace(
        text="hi",
        raw_text="hi",
        reasoning_content=None,
        tool_calls=None,
        finish_reason="stop",
        token_ids=[10, 11, 12],
        logprobs=None,
        weight_version_segments=[(0, 2), (2, 3)],
    )
    req = SimpleNamespace(prompt_token_ids=[1, 2], prompt="p", outputs=[comp])
    result = outputs_to_arrays(
        [req],
        input_ids=np.array([[1, 2]], dtype=np.int32),
        attention_mask=np.array([[1, 1]], dtype=np.int32),
        return_prompts=["p"],
        pad_token_id=0,
        max_seq_len=2,
        max_new_tokens=5,
        mesh=mesh,
    )
    assert result.completion_weight_versions is not None
    np.testing.assert_array_equal(np.asarray(result.completion_weight_versions), [[2, 2, 3, -1, -1]])
    assert result.completion_weight_versions.sharding == result.completion_ids.sharding
    assert result.completion_weight_versions.sharding == NamedSharding(mesh, PartitionSpec())


# ---------------------------------------------------------------------------
# #24: PPO _token_normalizer clamps global_count before dividing by GA
# ---------------------------------------------------------------------------


def test_pr306_24_ppo_token_normalizer_clamps_before_dividing_by_ga():
    mask = jnp.array([[1.0, 0.0, 0.0]])
    # 0 < global_count < GA: must NOT clamp post-division to 1.0.
    assert float(
        _token_normalizer(mask, jnp.asarray(2.0), token_level_loss=True, gradient_accumulation_steps=8)
    ) == pytest.approx(0.25)
    # global_count == 0: clamps global_count to 1.0 before dividing (denominator = 1/8, safe with 0 numerator).
    assert float(
        _token_normalizer(jnp.zeros_like(mask), jnp.asarray(0.0), token_level_loss=True, gradient_accumulation_steps=8)
    ) == pytest.approx(0.125)

    # End-to-end ppo_step invariance when 0 < num_positive_tokens < GA.
    class _TinyValueHead(spx.Module):
        def __init__(self):
            self.weight = spx.Parameter(jnp.ones((4, 1), dtype=jnp.float32) * 0.1)

        def forward(self, h):
            return h @ self.weight.value

    class _TinyPPO(spx.Module):
        def __init__(self, mesh):
            rng = np.random.default_rng(7)
            self.embed = spx.Parameter(jnp.asarray(rng.normal(size=(6, 4)), dtype=jnp.float32))
            self.head = spx.Parameter(jnp.asarray(rng.normal(size=(4, 6)), dtype=jnp.float32))
            self.value_head = _TinyValueHead()
            self.mesh = mesh

        def forward(self, input_ids, attention_mask, output_hidden_states=False, apply_lm_head=True):
            h = self.embed.value[input_ids]
            return SimpleNamespace(logits=h @ self.head.value if apply_lm_head else None, last_hidden_state=h)

    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    tx = optax.sgd(0.5)
    # 4 rows, GA=4, ONLY 2 positive completion tokens total (< GA=4)!
    input_ids = jnp.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 1], [4, 5, 1, 2]], dtype=jnp.int32)
    completion_mask = jnp.array([[1, 1], [1, 0], [1, 0], [1, 0]], dtype=jnp.float32)
    positive_mask = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)  # 2 positive tokens total < GA=4!
    batch = {
        "input_ids": input_ids,
        "attention_mask": jnp.ones_like(input_ids),
        "completion_mask": completion_mask,
        "old_logps": jnp.full((4, 2), -1.8, dtype=jnp.float32),
        "old_values": jnp.zeros((4, 2), dtype=jnp.float32),
        "advantages": jnp.array([[0.5, -0.3], [0.2, 0.0], [-0.4, 0.0], [0.1, 0.0]], dtype=jnp.float32),
        "returns": jnp.zeros((4, 2), dtype=jnp.float32),
        "positive_mask": positive_mask,
        "num_items_in_batch": jnp.sum(completion_mask),
        "num_positive_tokens": jnp.sum(completion_mask * positive_mask[:, None]),
    }
    assert float(batch["num_positive_tokens"]) == 2.0

    results = {}
    for ga in (1, 4):
        st = EasyDeLState.create(model=_TinyPPO(mesh))
        st = st.replace(tx=tx, opt_state=tx.init(st.graphstate))
        new_st, metrics = ppo_step(
            st,
            batch,
            prompt_length=2,
            cliprange=0.2,
            vf_coef=0.0,
            cliprange_value=0.2,
            entropy_coef=0.0,
            logprob_vocab_chunk_size=None,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=ga,
            is_training=True,
            token_level_loss=True,
            positive_nll_coef=0.5,
        )
        results[ga] = (new_st, metrics)

    np.testing.assert_allclose(
        float(results[4][1].other_metrics["positive_nll"]),
        float(results[1][1].other_metrics["positive_nll"]),
        rtol=1e-5,
        atol=1e-6,
    )
    for g1, g4 in zip(
        jax.tree_util.tree_leaves(results[1][0].graphstate),
        jax.tree_util.tree_leaves(results[4][0].graphstate),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(g4), np.asarray(g1), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# #25: Dynamic sampling zeroes difficulty_weights & refreshes discard metrics
# ---------------------------------------------------------------------------


def test_pr306_25_dynamic_sampling_zeroes_filtered_difficulty_weights_and_updates_discard_metrics():
    def make_scored(tag: int, rewards: list[float], *, discarded_groups: int = 0) -> _ScoredRollout:
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        rows = rewards_arr.size
        groups = rows // 2
        mb = {
            "prompt_ids": jnp.arange(groups * 2, dtype=jnp.int32).reshape(groups, 2),
            "prompt_mask": jnp.ones((groups, 2), dtype=jnp.int32),
            "completion_ids": jnp.full((rows, 2), tag, dtype=jnp.int32),
            "completion_mask": jnp.ones((rows, 2), dtype=jnp.int32),
            "ref_per_token_logps": jnp.full((rows, 2), -float(tag), dtype=jnp.float32),
            "advantages": jnp.ones((rows,), dtype=jnp.float32),
            "num_items_in_batch": jnp.asarray(rows * 2, dtype=jnp.int32),
        }
        metrics: dict[str, float | int | str] = {"generation_time": 1.0}
        if discarded_groups > 0:
            mb["difficulty_weights"] = jnp.asarray([0.0, 0.0] + [1.0] * (rows - 2), dtype=jnp.float32)
            mb["completion_mask"] = jnp.asarray([[0, 0], [0, 0]] + [[1, 1]] * (rows - 2), dtype=jnp.int32)
            metrics["rewards/discarded_groups"] = discarded_groups
            metrics["rewards/discarded_group_fraction"] = discarded_groups / groups
        return _ScoredRollout(
            model_batch=mb,
            metrics=metrics,
            rewards_per_func=jnp.asarray(rewards_arr[:, None]),
            reward_weights=jnp.ones((1,), dtype=jnp.float32),
        )

    # Part A: remaining filtered zero-variance group gets difficulty_weights == 0 (no sequence-loss dilution).
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(num_generations=2, dynamic_sampling=True, dynamic_sampling_max_attempts=2)
    trainer.scale_rewards = "group"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.steps_per_generation = 1
    trainer.num_iterations = 1
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_remaining = 0
    trainer.reward_funcs = []

    attempts = iter([make_scored(1, [0.0, 1.0, 2.0, 2.0]), make_scored(2, [5.0, 5.0, 3.0, 3.0])])
    trainer._generate_and_score_batch = lambda state, batch, is_train: next(attempts)
    mb, _ = trainer._preprocess_batch_input(None, {}, True)
    np.testing.assert_array_equal(np.asarray(mb["completion_mask"]).sum(-1), [2, 2, 0, 0])
    np.testing.assert_array_equal(np.asarray(mb["difficulty_weights"]), [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(mb["advantages"][2:]), [0.0, 0.0])

    # Confirm grpo_step with loss_type="grpo" gives the exact 1-active-group loss (not diluted by 1/2).
    cfg, model = _tiny_llama(seed=29)
    with cfg.mesh:
        st = EasyDeLState.create(model=model)
        full_metrics = grpo_step(st, mb, 2, 0.0, partition_spec=PartitionSpec(), is_training=False, loss_type="grpo")
        active_mb = {
            "prompt_ids": mb["prompt_ids"][:1],
            "prompt_mask": mb["prompt_mask"][:1],
            "completion_ids": mb["completion_ids"][:2],
            "completion_mask": mb["completion_mask"][:2],
            "ref_per_token_logps": mb["ref_per_token_logps"][:2],
            "advantages": mb["advantages"][:2],
        }
        active_metrics = grpo_step(
            st, active_mb, 2, 0.0, partition_spec=PartitionSpec(), is_training=False, loss_type="grpo"
        )
    np.testing.assert_allclose(float(full_metrics.loss), float(active_metrics.loss), rtol=1e-5, atol=1e-6)

    # Part B: resampled initially-discarded group updates rewards/discarded_* metrics to 0.
    trainer_b = object.__new__(GRPOTrainer)
    trainer_b.arguments = GRPOConfig(
        num_generations=2, dynamic_sampling=True, dynamic_sampling_max_attempts=2, discard_unscorable_groups=True
    )
    trainer_b.scale_rewards = "group"
    trainer_b.multi_objective_aggregation = "sum_then_normalize"
    trainer_b.steps_per_generation = 1
    trainer_b.num_iterations = 1
    trainer_b._buffered_grpo_batch = None
    trainer_b._buffered_grpo_remaining = 0
    trainer_b.reward_funcs = []

    attempts_b = iter([make_scored(1, [np.nan, 1.0, 0.0, 1.0], discarded_groups=1), make_scored(2, [0.0, 1.0, 4.0, 4.0])])
    trainer_b._generate_and_score_batch = lambda state, batch, is_train: next(attempts_b)
    _, metrics_b = trainer_b._preprocess_batch_input(None, {}, True)
    assert metrics_b["rewards/discarded_groups"] == 0
    assert metrics_b["rewards/discarded_group_fraction"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# #26: Toggle right-padding masks padded tokens when completion_mask is None
# ---------------------------------------------------------------------------


def test_pr306_26_toggle_right_pad_masks_padded_tokens_when_completion_mask_is_none():
    trainer = object.__new__(GRPOTrainer)
    trainer._pad_token_id = 0
    trainer._eos_token_id = [2]
    results = GenerationResults(
        generation_results=[],
        prompt_ids=jnp.array([[5, 6]], dtype=jnp.int32),
        prompt_mask=jnp.array([[1, 1]], dtype=jnp.int32),
        sequences=jnp.array([[5, 6, 10, 11, 12]], dtype=jnp.int32),
        completion_ids=jnp.array([[10, 11, 12]], dtype=jnp.int32),  # no EOS token (2)
        completion_mask=None,  # type: ignore[arg-type]
        decoded_prompts=["p"],
    )
    padded = trainer._pad_rollout_to_completion_width(results, width=6)
    np.testing.assert_array_equal(np.asarray(padded.completion_ids), [[10, 11, 12, 0, 0, 0]])
    assert padded.completion_mask is not None
    np.testing.assert_array_equal(np.asarray(padded.completion_mask), [[1, 1, 1, 0, 0, 0]])
    effective_mask = trainer._make_attn_mask(padded.completion_ids) * padded.completion_mask
    np.testing.assert_array_equal(np.asarray(effective_mask), [[1, 1, 1, 0, 0, 0]])


# ---------------------------------------------------------------------------
# #30: Chunked forward KL numerical stability (no ln(V) cancellation)
# ---------------------------------------------------------------------------


def test_pr306_30_chunked_forward_kl_no_cancellation_and_never_negative():
    rng = np.random.default_rng(30)
    B, L, H, V = 2, 9, 16, 1024
    T = 2.0
    h = rng.normal(size=(B, L, H)).astype(np.float32)
    w = (rng.normal(size=(H, V)) * 0.5).astype(np.float32)
    mask = np.ones((B, L), dtype=np.int32)

    # Identical student & teacher in bf16 -> KL must be exact 0.0 (never negative).
    loss_same, metrics_same = chunked_distillation_loss(
        student_hidden=jnp.asarray(h, jnp.bfloat16),
        teacher_hidden=jnp.asarray(h, jnp.bfloat16),
        student_lm_head_fn=lambda x: jnp.einsum("...h,hv->...v", x, jnp.asarray(w, jnp.bfloat16)),
        teacher_lm_head_fn=lambda x: jnp.einsum("...h,hv->...v", x, jnp.asarray(w, jnp.bfloat16)),
        loss_mask=jnp.asarray(mask),
        temperature=T,
        alpha=1.0,
        chunk_size=4,
    )
    assert float(loss_same) >= 0.0
    assert float(metrics_same["kl_loss"]) == pytest.approx(0.0, abs=1e-7)

    # Near-teacher + large constant logit offset (+1000): must match float64 reference closely.
    w_t = w + 0.02 * rng.normal(size=w.shape).astype(np.float32)
    hi = jax.lax.Precision.HIGHEST

    def run_near(ws):
        return chunked_distillation_loss(
            student_hidden=jnp.asarray(h),
            teacher_hidden=jnp.asarray(h),
            student_lm_head_fn=lambda x: jnp.einsum("...h,hv->...v", x, ws, precision=hi) + 1000.0,
            teacher_lm_head_fn=lambda x: jnp.einsum("...h,hv->...v", x, jnp.asarray(w_t), precision=hi),
            loss_mask=jnp.asarray(mask),
            temperature=T,
            alpha=1.0,
            chunk_size=4,
        )

    (loss_near, _), grad_near = jax.value_and_grad(run_near, has_aux=True)(jnp.asarray(w))

    hs64, ws64, wt64 = h.astype(np.float64)[:, :-1], w.astype(np.float64), w_t.astype(np.float64)
    m64 = mask.astype(np.float64)[:, 1:]
    ls64 = hs64 @ ws64 / T
    ls64 = ls64 - ls64.max(-1, keepdims=True)
    ls64 -= np.log(np.exp(ls64).sum(-1, keepdims=True))
    lt64 = hs64 @ wt64 / T
    lt64 = lt64 - lt64.max(-1, keepdims=True)
    lt64 -= np.log(np.exp(lt64).sum(-1, keepdims=True))
    ref_loss = float(((np.exp(lt64) * (lt64 - ls64)).sum(-1) * m64).sum() / m64.sum() * T**2)
    d_logits64 = T * (m64 / m64.sum())[..., None] * (np.exp(ls64) - np.exp(lt64))
    ref_grad = np.einsum("blh,blv->hv", hs64, d_logits64)

    np.testing.assert_allclose(float(loss_near), ref_loss, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(grad_near), ref_grad, rtol=1e-4, atol=1e-6)


def test_pr306_30_bregman_kl_large_spread_underflow_check_grads_and_memory():
    """Verify exact forward KL + VJP on large-spread / underflowing logits and no peak memory regression."""

    def _ref_kl_f64(t_vals, s_vals, temp: float) -> np.ndarray:
        t64 = np.asarray(t_vals, dtype=np.float64) / float(temp)
        s64 = np.asarray(s_vals, dtype=np.float64) / float(temp)
        t_lse = np.logaddexp.reduce(t64, axis=-1, keepdims=True)
        s_lse = np.logaddexp.reduce(s64, axis=-1, keepdims=True)
        lp = t64 - t_lse
        lq = s64 - s_lse
        p = np.exp(lp)
        q = np.exp(lq)
        with np.errstate(invalid="ignore", divide="ignore"):
            pr = np.where(p > 0.0, p * (lp - lq), 0.0)
        return np.sum(pr + (q - p), axis=-1)

    extreme_cases = [
        ([0.0, -120.0], [0.0, 0.0], 1.0, 1e-6, 1e-6),
        ([30.0, -30.0], [30.0, 25.0], 0.5, 1e-5, 1e-10),
        ([0.0, -150.0], [0.0, -40.0], 1.0, 1.0, 1e-12),
        ([0.0, 0.0], [0.0, -200.0], 1.0, 1e-5, 1e-5),
        ([0.0, -1e8], [0.0, -0.5], 1.0, 1e-5, 1e-6),
        ([0.0, -np.inf], [0.0, 0.0], 1.0, 1e-6, 1e-6),
        ([0.0, -np.inf], [0.0, -np.inf], 1.0, 1e-6, 1e-7),
    ]
    for tv, sv, temp, rtol, atol in extreme_cases:
        ta = jnp.asarray([tv], dtype=jnp.float32)
        sa = jnp.asarray([sv], dtype=jnp.float32)
        kl, ent = _per_token_kl_and_entropy(ta, sa, temp)
        ref = _ref_kl_f64([tv], [sv], temp)
        assert np.isfinite(float(kl[0])) and float(kl[0]) >= 0.0
        assert np.isfinite(float(ent[0])) and float(ent[0]) >= 0.0
        np.testing.assert_allclose(float(kl[0]), float(ref[0]), rtol=rtol, atol=atol)

    # check_grads in BOTH small-diff and large-spread regimes (default eps on the verifier's exact case).
    t_large = jnp.asarray([[30.0, -30.0, 20.0, -25.0]], dtype=jnp.float32)
    s_large = jnp.asarray([[30.0, 25.0, 20.0, -20.0]], dtype=jnp.float32)
    check_grads(lambda s_: jnp.sum(_per_token_kl_and_entropy(t_large, s_, 0.5)[0]), (s_large,), order=1, modes=["rev"])

    t_small = jnp.asarray([[1.0, -0.5, 2.0, -1.2]], dtype=jnp.float32)
    s_small = jnp.asarray([[1.01, -0.49, 1.99, -1.21]], dtype=jnp.float32)
    check_grads(lambda s_: jnp.sum(_per_token_kl_and_entropy(t_small, s_, 1.5)[0]), (s_small,), order=1, modes=["rev"])

    t_huge = jnp.asarray([[0.0, -120.0, -5.0, 1.0]], dtype=jnp.float32)
    s_huge = jnp.asarray([[0.0, 0.0, -2.0, 0.5]], dtype=jnp.float32)
    check_grads(
        lambda s_: jnp.sum(_per_token_kl_and_entropy(t_huge, s_, 1.0)[0]),
        (s_huge,),
        order=1,
        modes=["rev"],
        eps=1e-3,
    )

    # Mixed dtypes: teacher bf16, student f32 -> zero teacher grad preserves bf16 dtype, student grad stays f32.
    gt, gs = jax.grad(lambda t_, s_: jnp.sum(_per_token_kl_and_entropy(t_, s_, 1.0)[0]), argnums=(0, 1))(
        jnp.asarray([[1.0, 2.0, -1.0]], dtype=jnp.bfloat16),
        jnp.asarray([[1.2, 1.8, -0.5]], dtype=jnp.float32),
    )
    assert gt.dtype == jnp.bfloat16
    assert gs.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(gt, dtype=np.float32), 0.0)
    assert np.all(np.isfinite(np.asarray(gs)))

    # XLA peak temp memory regression guard vs dea90fa baseline
    # (standalone <= 2x fwd / 4x ckpt_vag; chunked_distillation_loss <= 3.1x vag vs 3.51x baseline).
    B_m, V_m = 64, 32768
    unit = B_m * V_m * 4
    t_bf = jnp.ones((B_m, V_m), dtype=jnp.bfloat16)
    s_bf = jnp.ones((B_m, V_m), dtype=jnp.bfloat16)
    fwd_bytes = (
        jax.jit(lambda x, y: jnp.sum(_per_token_kl_and_entropy(x, y, 1.0)[0]))
        .lower(t_bf, s_bf)
        .compile()
        .memory_analysis()
        .temp_size_in_bytes
    )
    ckpt_bytes = (
        jax.jit(
            jax.value_and_grad(
                lambda x, y: jnp.sum(jax.checkpoint(lambda a, b: _per_token_kl_and_entropy(a, b, 1.0)[0])(x, y)),
                argnums=1,
            )
        )
        .lower(t_bf, s_bf)
        .compile()
        .memory_analysis()
        .temp_size_in_bytes
    )
    assert fwd_bytes / unit <= 2.1
    assert ckpt_bytes / unit <= 4.1

    B_c, L_c, H_c, C_c = 4, 65, 64, 16
    unit_c = B_c * C_c * V_m * 4
    sh = jnp.ones((B_c, L_c, H_c), dtype=jnp.bfloat16)
    th = jnp.ones((B_c, L_c, H_c), dtype=jnp.bfloat16)
    sw = jnp.ones((H_c, V_m), dtype=jnp.bfloat16)
    tw = jnp.ones((H_c, V_m), dtype=jnp.bfloat16)
    m_c = jnp.ones((B_c, L_c), dtype=jnp.int32)

    def _chunk_loss(sh_, sw_, th_, tw_, m_):
        return chunked_distillation_loss(
            student_hidden=sh_,
            teacher_hidden=th_,
            student_lm_head_fn=lambda h_: h_ @ sw_,
            teacher_lm_head_fn=lambda h_: h_ @ tw_,
            attention_mask=m_,
            loss_mask=m_,
            labels=None,
            temperature=1.0,
            alpha=1.0,
            chunk_size=C_c,
        )[0]

    chunk_fwd_bytes = (
        jax.jit(_chunk_loss)
        .lower(sh, sw, th, tw, m_c)
        .compile()
        .memory_analysis()
        .temp_size_in_bytes
    )
    chunk_vag_bytes = (
        jax.jit(jax.value_and_grad(_chunk_loss, argnums=(0, 1)))
        .lower(sh, sw, th, tw, m_c)
        .compile()
        .memory_analysis()
        .temp_size_in_bytes
    )
    assert chunk_fwd_bytes / unit_c <= 2.1
    assert chunk_vag_bytes / unit_c <= 3.1


# ---------------------------------------------------------------------------
# #31: GMPO + top_entropy_quantile < 1.0 masks low-entropy token gradients
# ---------------------------------------------------------------------------


def test_pr306_31_gmpo_top_entropy_quantile_masks_low_entropy_token_gradients():
    # Direct unit test on _compute_grpo_policy_loss_terms with entropy_mask.
    old_logps = jnp.array([[-1.0, -1.0, -1.0, -1.0]], dtype=jnp.float32)
    advantages = jnp.array([[1.5]], dtype=jnp.float32)
    completion_mask = jnp.ones((1, 4), dtype=jnp.float32)
    entropy_mask = jnp.array([[1.0, 0.0, 1.0, 0.0]], dtype=jnp.float32)

    def gmpo_loss(logps):
        per_token, _ = _compute_grpo_policy_loss_terms(
            per_token_logps=logps,
            old_per_token_logps=old_logps,
            advantages=advantages,
            completion_mask=completion_mask,
            loss_type="gmpo",
            epsilon=0.2,
            epsilon_high=0.2,
            delta=None,
            importance_sampling_level="token",
            sapo_temperature_pos=1.0,
            sapo_temperature_neg=1.05,
            vespo_k_pos=2.0,
            vespo_lambda_pos=3.0,
            vespo_k_neg=3.0,
            vespo_lambda_neg=2.0,
            entropy_mask=entropy_mask,
        )
        return jnp.sum(per_token * entropy_mask) / jnp.sum(completion_mask)

    grad_logps = jax.grad(gmpo_loss)(jnp.array([[-0.95, -0.95, -1.05, -1.05]], dtype=jnp.float32))
    assert float(grad_logps[0, 1]) == 0.0
    assert float(grad_logps[0, 3]) == 0.0
    assert abs(float(grad_logps[0, 0])) > 1e-4
    assert abs(float(grad_logps[0, 2])) > 1e-4

    # Full grpo_step test with a tiny model whose positions have cleanly distinct entropies:
    # positions 0, 2 have flat (high-entropy) logits; positions 1, 3 have peaked (low-entropy) logits.
    class _PerPosLogitModel:
        def __init__(self, logits_table, mesh):
            self.logits_table = logits_table
            self.mesh = mesh

        def __call__(self, input_ids, attention_mask=None, **kwargs):
            del attention_mask, kwargs
            return SimpleNamespace(logits=jnp.broadcast_to(self.logits_table, (*input_ids.shape, self.logits_table.shape[-1])))

    @jax.tree_util.register_pytree_node_class
    class _LogitState:
        def __init__(self, graphstate, mesh):
            self.graphstate = graphstate
            self.mesh = mesh
            self.step = jnp.asarray(0)

        def tree_flatten(self):
            return (self.graphstate,), (self.mesh,)

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(children[0], aux[0])

        @property
        def model(self):
            return _PerPosLogitModel(self.graphstate, self.mesh)

        def merge(self, tree):
            return _PerPosLogitModel(tree, self.mesh)

    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    # Seq len = 1 prompt + 4 completion = 5; logits[:, 0:4] predict completion positions 0..3.
    logits_init = jnp.array(
        [
            [
                [0.1, 0.0, -0.1, 0.0],  # pos 0: high entropy (~1.38)
                [8.0, 0.0, 0.0, 0.0],  # pos 1: low entropy (~0.009)
                [0.0, 0.1, 0.0, -0.1],  # pos 2: high entropy (~1.38)
                [0.0, 8.0, 0.0, 0.0],  # pos 3: low entropy (~0.009)
                [0.0, 0.0, 0.0, 0.0],  # unused final pos
            ]
        ],
        dtype=jnp.float32,
    )
    batch = {
        "prompt_ids": jnp.array([[0]], dtype=jnp.int32),
        "prompt_mask": jnp.array([[1]], dtype=jnp.int32),
        "completion_ids": jnp.array([[0, 1, 2, 0]], dtype=jnp.int32),
        "completion_mask": jnp.ones((1, 4), dtype=jnp.float32),
        "advantages": jnp.array([1.2], dtype=jnp.float32),
    }

    def step_loss(table):
        st = _LogitState(table, mesh)
        metrics = grpo_step(
            st,
            batch,
            1,
            0.0,
            partition_spec=PartitionSpec(),
            is_training=False,
            loss_type="gmpo",
            epsilon=0.2,
            epsilon_high=0.2,
            top_entropy_quantile=0.5,
        )
        return metrics.loss

    grad_table = np.asarray(jax.grad(step_loss)(logits_init))[0]
    # Low-entropy completion positions 1 and 3 must receive EXACTLY zero gradient.
    np.testing.assert_array_equal(grad_table[1], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(grad_table[3], np.zeros(4, dtype=np.float32))
    # High-entropy completion positions 0 and 2 must receive non-zero gradient.
    assert np.max(np.abs(grad_table[0])) > 1e-4
    assert np.max(np.abs(grad_table[2])) > 1e-4


# ---------------------------------------------------------------------------
# #37: GFPO _select_rollout_indices preserves unscorable group discarding
# ---------------------------------------------------------------------------


def test_pr306_37_gfpo_selection_preserves_unscorable_group_discard():
    """Dropping the NaN row from an unscorable group during GFPO selection must NOT un-discard that group."""

    class _GFPOStubTrainer(GRPOTrainer):
        def _select_rollout_indices(self, completion_ids, completion_mask, rewards, generation_factor):
            # Keep first 2 of 4 rows per group -> drops index 2 (the NaN row!) from group 0.
            del completion_ids, completion_mask, rewards, generation_factor
            return jnp.array([0, 1, 4, 5], dtype=jnp.int32)

    trainer = object.__new__(_GFPOStubTrainer)
    trainer.arguments = GRPOConfig(
        num_generations=4,
        beta=0.0,
        discard_unscorable_groups=True,
        mask_truncated_completions=False,
        log_completions=False,
    )
    trainer._pad_token_id = 0
    trainer._eos_token_id = [2]
    trainer.ref_state = None
    trainer.ref_logps_chunk_size = None
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.environment_factory = None
    trainer.scale_rewards = "group"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer._group_reduction = "mean"
    trainer.processing_class = None

    def reward_fn(completions, **kwargs):
        del completions, kwargs
        # Group 0 has a None (NaN) at row 2; Group 1 is fully scorable.
        return [1.0, 2.0, None, 3.0, 0.0, 1.0, 2.0, 3.0]

    trainer.reward_funcs = [reward_fn]
    trainer.reward_func_names = ["score"]
    trainer.reward_processing_classes = [None]
    trainer.reward_weights = jnp.ones((1,), dtype=jnp.float32)

    trainer._apply_user_data_collator = lambda b: b
    trainer._extract_reward_batch_sidechannels = lambda b: {}
    trainer._purify_batch = lambda b: b
    trainer._all_gather = lambda x: x
    trainer._maybe_log_grpo_completions = lambda **kwargs: []
    trainer._log_training_generations_to_wandb = lambda **kwargs: None
    trainer._warn_on_ungradeable_completions = lambda **kwargs: None
    trainer._maybe_run_tool_call_generation_loop = lambda state, results, completion_mask: (results, completion_mask)
    trainer._should_compute_old_per_token_logps = lambda **kwargs: False

    prompt_ids = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    prompt_mask = jnp.ones((2, 2), dtype=jnp.int32)
    completion_ids = jnp.full((8, 3), 5, dtype=jnp.int32)
    completion_mask = jnp.ones((8, 3), dtype=jnp.int32)
    fake_results = GenerationResults(
        generation_results=[],
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        sequences=jnp.concatenate([prompt_ids.repeat(4, axis=0), completion_ids], axis=-1),
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        decoded_prompts=["p0", "p1"],
        completion_prompts=["p0"] * 4 + ["p1"] * 4,
        text=[f"c{i}" for i in range(8)],
        raw_text=[f"c{i}" for i in range(8)],
    )
    trainer.rollout = lambda **kwargs: fake_results

    fake_state = SimpleNamespace(model=SimpleNamespace(forward=lambda input_ids=None, attention_mask=None: None), step=0)
    scored = trainer._generate_and_score_batch(
        fake_state,  # type: ignore[arg-type]
        {"input_ids": prompt_ids, "attention_mask": prompt_mask},
        is_train=True,
    )

    assert scored.metrics["rewards/discarded_groups"] == 1
    assert scored.metrics["rewards/discarded_group_fraction"] == pytest.approx(0.5)
    # Group 0 (rows 0, 1 after selection) must stay completely discarded while group 1 (rows 2, 3) stays active.
    np.testing.assert_array_equal(np.asarray(scored.model_batch["completion_mask"][:2]), 0)
    np.testing.assert_array_equal(np.asarray(scored.model_batch["completion_mask"][2:]), 1)
    np.testing.assert_array_equal(np.asarray(scored.model_batch["advantages"][:2]), 0.0)
    assert float(jnp.min(jnp.abs(scored.model_batch["advantages"][2:]))) > 0.5
    np.testing.assert_array_equal(np.asarray(scored.model_batch["difficulty_weights"]), [0.0, 0.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# #45: validate_prefix_shared_model gaps
# ---------------------------------------------------------------------------


def _prefix_stub(*, mesh=None, **overrides):
    values = dict(attn_mechanism="vanilla", kv_cache_quantization_config=None, sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    values.update(overrides)
    return SimpleNamespace(config=SimpleNamespace(**values), mesh=mesh)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"sharding_axis_dims": {"pp": 2, "fsdp": -1}}, "pipeline"),
        ({"mesh": SimpleNamespace(shape={"pp": 2, "dp": 1}), "sharding_axis_dims": (-1, 1, 1, 1, 1, 1)}, "pipeline"),
        ({"model_type": "mamba"}, "mamba"),
        ({"model_type": "mamba2"}, "mamba2"),
        ({"model_type": "falcon_mamba"}, "falcon_mamba"),
        ({"model_type": "mosaic_mpt"}, "mosaic_mpt"),
        ({"model_type": "mpt"}, "mpt"),
        ({"model_type": "bloom"}, "bloom"),
        ({"alibi": True}, "ALiBi"),
        ({"attn_config": SimpleNamespace(alibi=True, attn_pdrop=0.0)}, "ALiBi"),
        ({"attn_pdrop": 0.1}, "dropout"),
        ({"resid_pdrop": 0.1}, "dropout"),
        ({"embd_pdrop": 0.1}, "dropout"),
        ({"hidden_dropout": 0.1}, "dropout"),
        ({"attn_config": SimpleNamespace(alibi=False, attn_pdrop=0.1)}, "dropout"),
    ],
)
def test_pr306_45_validate_prefix_shared_model_rejects_gaps(kwargs, match):
    with pytest.raises(NotImplementedError, match=match):
        validate_prefix_shared_model(_prefix_stub(**kwargs))


def test_pr306_45_validate_prefix_shared_model_accepts_dict_pp1_and_qwen3_next():
    validate_prefix_shared_model(_prefix_stub(sharding_axis_dims={"pp": 1, "fsdp": -1}))
    # 5-axis sharding_axis_names without "pp": dp=2 at index 0 must NOT be misread as pp=2.
    validate_prefix_shared_model(
        _prefix_stub(
            sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
            sharding_axis_dims=(2, 1, 1, 1, 1),
        )
    )
    # Explicit "pp" at non-zero index with pp > 1 must still be rejected.
    with pytest.raises(NotImplementedError, match="pipeline"):
        validate_prefix_shared_model(
            _prefix_stub(
                sharding_axis_names=("dp", "pp", "fsdp", "ep", "tp", "sp"),
                sharding_axis_dims=(1, 2, 1, 1, 1, 1),
            )
        )
    validate_prefix_shared_model(
        _prefix_stub(
            model_type="qwen3_next",
            layer_types=["full_attention", "linear_attention"],
            mesh=SimpleNamespace(shape={"pp": 1, "dp": 1}),
        )
    )


# ---------------------------------------------------------------------------
# #47: Replay buffer stores behavior_per_token_logps on host and restores jax.Array
# ---------------------------------------------------------------------------


def test_pr306_47_replay_buffer_stores_behavior_logps_on_host_and_restores_jax_array():
    trainer = object.__new__(GRPOWithReplayBufferTrainer)
    trainer.arguments = GRPOConfig(num_generations=2, esurge_importance_sampling_mode="token_truncate")
    trainer.num_generations = 2
    trainer._pad_token_id = 0
    trainer.replay_buffer = _ReplayBuffer(max_size=4, seed=0)

    behavior_logps = jnp.array([[-1.0, -2.0], [-1.5, -2.5], [-9.0, -9.0], [-9.0, -9.0]], dtype=jnp.float32)
    batch = {
        "prompt_ids": jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
        "prompt_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "completion_ids": jnp.array([[5, 6], [7, 8], [9, 9], [9, 9]], dtype=jnp.int32),
        "completion_mask": jnp.ones((4, 2), dtype=jnp.int32),
        "advantages": jnp.array([-1.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
        "behavior_per_token_logps": behavior_logps,
    }
    updated, replay_metrics = trainer._apply_replay_buffer(batch)
    assert replay_metrics["replay_groups_added"] == 1
    assert replay_metrics["replay_groups_replaced"] == 1

    # Stored buffer item must be a host NumPy array (never pinning accelerator HBM).
    stored_group = trainer.replay_buffer._heap[0][2]
    assert isinstance(stored_group["behavior_per_token_logps"], np.ndarray)
    assert not isinstance(stored_group["behavior_per_token_logps"], jax.Array)

    # Sampled batch must restore jax.Array with group 0's behavior logps copied into group 1.
    assert isinstance(updated["behavior_per_token_logps"], jax.Array)
    np.testing.assert_allclose(np.asarray(updated["behavior_per_token_logps"][2:4]), [[-1.0, -2.0], [-1.5, -2.5]])


# ---------------------------------------------------------------------------
# #53: OnPolicyDistillationConfig trust_region async guard & temperature override
# ---------------------------------------------------------------------------


def test_pr306_53_opd_trust_region_rejects_async_and_honors_extra_kwargs_temperature():
    with pytest.raises(ValueError, match="async_mode"):
        OnPolicyDistillationConfig(distillation_mode="trust_region", async_mode=True)

    # Divergence mode still allows async_mode=True.
    div_cfg = OnPolicyDistillationConfig(distillation_mode="divergence", async_mode=True)
    assert div_cfg.async_mode is True

    # generation_extra_kwargs["temperature"] overrides temperature_sampling / generation_temperature.
    cfg_extra = OnPolicyDistillationConfig(
        distillation_mode="trust_region",
        temperature_sampling=0.7,
        generation_temperature=0.8,
        generation_extra_kwargs={"temperature": 1.25},
    )
    assert cfg_extra.trust_region_sampling_temperature == pytest.approx(1.25)

    # generation_config_overrides["temperature"] takes highest precedence.
    cfg_override = OnPolicyDistillationConfig(
        distillation_mode="trust_region",
        temperature_sampling=0.7,
        generation_extra_kwargs={"temperature": 1.25},
        generation_config_overrides={"temperature": 1.5},
    )
    assert cfg_override.trust_region_sampling_temperature == pytest.approx(1.5)

    # Non-positive temperature or do_sample=False in generation_extra_kwargs must be rejected.
    with pytest.raises(ValueError, match="positive sampling temperature"):
        OnPolicyDistillationConfig(
            distillation_mode="trust_region",
            generation_extra_kwargs={"temperature": 0.0},
        )
    with pytest.raises(ValueError, match="generation_do_sample"):
        OnPolicyDistillationConfig(
            distillation_mode="trust_region",
            generation_extra_kwargs={"do_sample": False},
        )


# ---------------------------------------------------------------------------
# Bonus (#301): GRPOTrainer.compute_refmodel_logps positional static_argnums
#               & _call_compute_refmodel_logps under real compiled in_shardings
# ---------------------------------------------------------------------------


def test_pr306_301_grpo_compiled_compute_refmodel_logps_and_positional_dispatch(tmp_path):
    """Real compiled GRPOTrainer.compute_refmodel_logps must work with in_shardings + dynamic prompt_length."""
    from datasets import Dataset

    _, policy = _tiny_llama(seed=11)
    _, ref_model = _tiny_llama(seed=12)
    ref_model.eval()

    class _MinimalTok:
        pad_token_id = 0
        eos_token_id = 2
        padding_side = "left"

        def __call__(self, text, **kwargs):
            del kwargs
            items = [text] if isinstance(text, str) else list(text)
            return {"input_ids": [[1, 3, 4] for _ in items], "attention_mask": [[1, 1, 1] for _ in items]}

        def batch_decode(self, ids, **kwargs):
            del kwargs
            return ["ok" for _ in range(len(ids))]

        def decode(self, ids, **kwargs):
            del ids, kwargs
            return "ok"

    prompt_ds = Dataset.from_dict({"prompt": ["hello world", "foo bar"]})
    trainer = GRPOTrainer(
        arguments=GRPOConfig(
            max_training_steps=1,
            total_batch_size=2,
            num_generations=2,
            beta=0.04,
            max_prompt_length=6,
            max_completion_length=4,
            use_wandb=False,
            save_directory=str(tmp_path / "grpo_ref_logps"),
        ),
        model=policy,
        reward_funcs=lambda **kw: [0.0] * len(kw.get("completions", [])),
        train_dataset=prompt_ds,
        processing_class=_MinimalTok(),
    )
    assert trainer.ref_state is not None
    assert trainer.compute_refmodel_logps is not None

    # Seq len = 7 with dynamic prompt_length = 3 != max_prompt_length (6).
    ids = jnp.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    mask = jnp.ones_like(ids)
    empty_kw = normalize_generation_model_kwargs(None)

    expected = get_per_token_logps(trainer.ref_state.model, ids, mask, 3, model_kwargs=empty_kw)

    # 1. Via _call_compute_refmodel_logps(..., prompt_length=3)
    got_via_helper = trainer._call_compute_refmodel_logps(
        trainer.ref_state.graphstate,
        trainer.ref_state.graphother,
        ids,
        mask,
        empty_kw,
        prompt_length=3,
    )
    assert got_via_helper.shape == (2, 4)
    np.testing.assert_allclose(np.asarray(got_via_helper), np.asarray(expected), atol=1e-5)

    # 2. Direct call with keyword prompt_length=3 and positional 3
    got_kw = trainer.compute_refmodel_logps(
        trainer.ref_state.graphstate,
        trainer.ref_state.graphother,
        ids,
        mask,
        empty_kw,
        prompt_length=3,
    )
    got_pos = trainer.compute_refmodel_logps(
        trainer.ref_state.graphstate,
        trainer.ref_state.graphother,
        ids,
        mask,
        empty_kw,
        3,
    )
    np.testing.assert_allclose(np.asarray(got_kw), np.asarray(expected), atol=1e-5)
    np.testing.assert_allclose(np.asarray(got_pos), np.asarray(expected), atol=1e-5)

    # 3. Legacy 5-arg call with model_kwargs=None defaults to max_prompt_length=6 -> (2, 1)
    got_default = trainer.compute_refmodel_logps(
        trainer.ref_state.graphstate,
        trainer.ref_state.graphother,
        ids,
        mask,
        None,
    )
    assert got_default.shape == (2, 1)

    # 4. Direct compile_trainer_step(..., static_argnums=(5,), in_shardings=...) stub called via
    #    _call_compute_refmodel_logps MUST dispatch positionally (never triggering pjit kwarg ValueError).
    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    rep = NamedSharding(mesh, PartitionSpec())
    kw_shardings = {k: None for k in empty_kw.keys()}

    def _direct_compiled_fn(gs, go, seq_ids, seq_mask, model_kwargs, prompt_length):
        del gs, go, seq_mask, model_kwargs
        return -0.75 * jnp.ones((seq_ids.shape[0], seq_ids.shape[1] - int(prompt_length)), dtype=jnp.float32)

    stub = object.__new__(GRPOTrainer)
    stub.compute_refmodel_logps = compile_trainer_step(
        _direct_compiled_fn,
        mesh=mesh,
        static_argnums=(5,),
        in_shardings=(rep, rep, rep, rep, kw_shardings),
        out_shardings=rep,
    )
    out_direct = stub._call_compute_refmodel_logps(
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        ids,
        mask,
        empty_kw,
        prompt_length=2,
    )
    assert out_direct.shape == (2, 5)
    np.testing.assert_allclose(np.asarray(out_direct), -0.75)

    # 5. Legacy 5-arg stub, keyword-only stub, and legacy keyword-only pjit fallback all stay compatible.
    stub.compute_refmodel_logps = lambda gs, go, s_ids, s_mask, kw: jnp.full((s_ids.shape[0], 1), -0.25)
    out_legacy5 = stub._call_compute_refmodel_logps(0, 0, ids, mask, empty_kw, prompt_length=2)
    assert out_legacy5.shape == (2, 1)

    def _kw_only_stub(gs, go, s_ids, s_mask, kw, *, prompt_length=None):
        del gs, go, s_mask, kw
        return jnp.full((s_ids.shape[0], s_ids.shape[1] - int(prompt_length)), -0.5)

    stub.compute_refmodel_logps = _kw_only_stub
    out_kw_only = stub._call_compute_refmodel_logps(0, 0, ids, mask, empty_kw, prompt_length=4)
    assert out_kw_only.shape == (2, 3)

    def _legacy_pjit_kw_fn(gs, go, s_ids, s_mask, kw=None, *, graphdef=None, prompt_length=None):
        del gs, go, s_mask, kw, graphdef
        pl = 6 if prompt_length is None else int(prompt_length)
        return jnp.full((s_ids.shape[0], s_ids.shape[1] - pl), -0.1)

    stub.compute_refmodel_logps = compile_trainer_step(
        functools.partial(_legacy_pjit_kw_fn, graphdef="gd"),
        mesh=mesh,
        static_argnames=("graphdef", "prompt_length"),
        in_shardings=(rep, rep, rep, rep, kw_shardings),
        out_shardings=rep,
    )
    out_legacy_pjit = stub._call_compute_refmodel_logps(
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        ids,
        mask,
        empty_kw,
        prompt_length=3,
    )
    assert out_legacy_pjit.shape == (2, 1)
    np.testing.assert_allclose(np.asarray(out_legacy_pjit), -0.1)
