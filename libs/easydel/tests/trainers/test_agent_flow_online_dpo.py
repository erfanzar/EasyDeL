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
"""OnlineDPO as an AgentFlow inner trainer: explicit DPO loss masks and per-group preference pairs.

Tests marked ``cpu_ok`` do no array computation. The others run the production DPO
``concatenated_forward`` / ``compute_dpo_losses`` and must run on an accelerator; JAX is
imported inside those tests only.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

VOCAB = 128
BETA = 0.1


class _DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        ids = [ord(c) for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids if int(i) > 0)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + (
            "<assistant>" if add_generation_prompt else ""
        )


def _make_env_cls():
    from easydel.environments._base import AgenticEnvironment, ResetResult, StepResult

    class _GEnv(AgenticEnvironment):
        """Two-turn env: each step's reward counts the ``g`` characters in the action."""

        def __init__(self, **kwargs):
            self._turn = 0

        def reset(self, seed: int | None = None) -> ResetResult:
            self._turn = 0
            return ResetResult(observation="start", info={})

        def step(self, action: str) -> StepResult:
            self._turn += 1
            if "FAIL" in action:
                return StepResult(observation="", reward=0.0, terminated=True, truncated=False, info={"env_error": True})
            done = self._turn >= 2
            return StepResult(
                observation="" if done else "obs1",
                reward=float(action.count("g")),
                terminated=done,
                truncated=False,
                info={},
            )

    return _GEnv


def _make_flow_and_batch():
    """Run 3 groups x 2 episodes through AgentFlow with an OnlineDPO inner trainer.

    Group 0: rewards 3 vs 0 -> pair. Group 1: 1 vs 1 -> tie. Group 2: one env_error -> insufficient.
    """
    from easydel.trainers.agent_flow_trainer import AgentFlowConfig, AgentFlowTrainer
    from easydel.trainers.online_dpo_trainer import OnlineDPOTrainer

    inner = object.__new__(OnlineDPOTrainer)
    inner.arguments = SimpleNamespace(async_mode=False, max_prompt_length=16, max_completion_length=40)
    inner.processing_class = _DummyTokenizer()
    inner._preprocess_batch_input = lambda state, batch, is_train: ({}, {})

    actions = iter(["g", "gg", "x", "x", "g", "x", "x", "g", "g", "g", "FAIL"])

    def gen_turn(state, prompt_ids, prompt_text, turn_idx):
        text = next(actions)
        return [ord(c) for c in text], text, None

    env_cls = _make_env_cls()
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=2, group_size=2, max_workers=1, reward_mode="sum"),
        trainer=inner,
        environment_factory=lambda **kw: env_cls(),
        generate_turn_fn=gen_turn,
    )
    batch, metrics = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p0", "p1", "p2"]}, True)
    return flow, batch, metrics


class _RowPositionLogitModel:
    """Toy causal LM whose logits for row ``b`` at position ``t`` are a free parameter ``bias[b, t]``.

    Every (row, position) owns its own parameters, so a token's gradient cannot cancel against
    another row's and a zero gradient means that token was not scored.
    """

    def __init__(self, bias):
        self.bias = bias

    def forward(self, input_ids, attention_mask):
        assert tuple(input_ids.shape) == tuple(self.bias.shape[:2])
        return SimpleNamespace(logits=self.bias, last_hidden_state=None)

    __call__ = forward


def _forward(bias, batch):
    from easydel.trainers.direct_preference_optimization_trainer._fn import concatenated_forward

    return concatenated_forward(
        _RowPositionLogitModel(bias),
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )


def _scored_positions(batch, *, use_loss_mask: bool) -> np.ndarray:
    """``[2N, L]`` bool: logit positions (chosen rows then rejected rows) that score a completion token."""
    prompt_w = batch["prompt_input_ids"].shape[1]
    comp_w = batch["chosen_input_ids"].shape[1]
    rows = []
    for side in ("chosen", "rejected"):
        scored = np.asarray(batch[f"{side}_attention_mask"]).astype(bool)
        if use_loss_mask:
            scored = scored & np.asarray(batch[f"{side}_loss_mask"]).astype(bool)
        # The token at sequence position ``prompt_w + j`` is predicted by the logits at ``prompt_w + j - 1``.
        pos = np.zeros((scored.shape[0], prompt_w + comp_w), dtype=bool)
        pos[:, prompt_w - 1 : prompt_w + comp_w - 1] = scored
        rows.append(pos)
    return np.concatenate(rows, axis=0)


def _numpy_logps(bias, batch, *, use_loss_mask: bool):
    """Independent reference: sum log-softmax(bias[b, t])[token t+1] over scored completion tokens."""
    bias = np.asarray(bias, dtype=np.float64)
    shift = bias.max(-1, keepdims=True)
    log_probs = bias - shift - np.log(np.exp(bias - shift).sum(-1, keepdims=True))
    scored = _scored_positions(batch, use_loss_mask=use_loss_mask)
    n = batch["prompt_input_ids"].shape[0]
    prompt = np.asarray(batch["prompt_input_ids"])
    out = {}
    for offset, side in ((0, "chosen"), (n, "rejected")):
        comp = np.asarray(batch[f"{side}_input_ids"])
        sums = []
        for row in range(n):
            seq = np.concatenate([prompt[row], comp[row]])
            total = 0.0
            for t in np.nonzero(scored[offset + row])[0]:
                total += log_probs[offset + row, t, seq[t + 1]]
            sums.append(total)
        out[side] = np.asarray(sums)
    return out


def _numpy_sigmoid_dpo(chosen, rejected, ref_chosen, ref_rejected):
    z = BETA * ((chosen - ref_chosen) - (rejected - ref_rejected))
    return np.log1p(np.exp(-z))


@pytest.mark.cpu_ok
def test_select_group_preference_pairs_handles_ties_and_insufficient_valid():
    from easydel.trainers.agent_flow_trainer.agent_flow_trainer import select_group_preference_pairs

    rewards = [0.5, 2.0, -1.0, 1.0, 1.0, 1.0, 3.0, 9.0, 0.0, 4.0, float("nan"), 2.0]
    valid = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    sel = select_group_preference_pairs(rewards, valid, 3)
    # g0: max 2.0 (idx 1) vs min -1.0 (idx 2); g1: all 1.0 -> tie; g2: one valid -> insufficient;
    # g3: NaN dropped, 4.0 (idx 9) vs 2.0 (idx 11).
    assert sel.pairs == ((1, 2), None, None, (9, 11))
    assert sel.num_pairs == 2
    assert sel.skipped_ties == 1
    assert sel.skipped_insufficient == 1

    # Partial ties keep the first-reaching max/min; equal max+min across all candidates is a tie.
    assert select_group_preference_pairs([1.0, 5.0, 5.0, 1.0], [1, 1, 1, 1], 4).pairs == ((1, 0),)
    assert select_group_preference_pairs([2.0, 2.0], [1, 1], 2).skipped_ties == 1
    assert select_group_preference_pairs([2.0, 1.0], [0, 0], 2).skipped_insufficient == 1
    with pytest.raises(ValueError, match="groups of"):
        select_group_preference_pairs([1.0, 2.0, 3.0], [1, 1, 1], 2)


@pytest.mark.cpu_ok
def test_online_dpo_inner_trainer_is_accepted_and_paired_groups_have_two_episodes():
    from easydel.trainers.agent_flow_trainer import AgentFlowConfig, AgentFlowTrainer
    from easydel.trainers.online_dpo_trainer import OnlineDPOTrainer

    inner = object.__new__(OnlineDPOTrainer)
    inner.arguments = SimpleNamespace(async_mode=False, max_prompt_length=8, max_completion_length=8)
    inner.processing_class = _DummyTokenizer()
    inner._preprocess_batch_input = lambda state, batch, is_train: ({}, {})
    env_cls = _make_env_cls()
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1), trainer=inner, environment_factory=lambda **kw: env_cls()
    )
    assert flow.trainer is inner
    assert flow.arguments.base_trainer_type == "online_dpo"
    # Pairs need >= 2 episodes per group in training and in evaluation.
    assert flow._resolve_group_size(True) == 2
    assert flow._resolve_group_size(False) == 2


def test_agent_flow_builds_one_masked_pair_per_group_in_dpo_layout():
    _, batch, metrics = _make_flow_and_batch()
    expected_keys = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_loss_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_loss_mask",
    }
    assert set(batch) == expected_keys
    assert batch["prompt_input_ids"].shape == (3, 16)
    for side in ("chosen", "rejected"):
        for suffix in ("input_ids", "attention_mask", "loss_mask"):
            assert batch[f"{side}_{suffix}"].shape == (3, 40)

    assert metrics["agent_flow/dpo_pairs"] == 1.0
    assert metrics["agent_flow/dpo_groups"] == 3.0
    assert metrics["agent_flow/dpo_skipped_ties"] == 1.0
    assert metrics["agent_flow/dpo_skipped_insufficient_valid"] == 1.0
    assert metrics["agent_flow/dpo_skipped_prompt_mismatch"] == 0.0
    assert metrics["agent_flow/dpo_reward_margin"] == 3.0

    def _text(side, mask_key, row=0):
        ids = np.asarray(batch[f"{side}_input_ids"][row])
        mask = np.asarray(batch[mask_key][row]).astype(bool)
        return "".join(chr(int(t)) for t in ids[mask])

    # Chosen = reward-3 episode ("g" then "gg"), rejected = reward-0 episode ("x" then "x").
    assert _text("chosen", "chosen_loss_mask") == "ggg"
    assert _text("rejected", "rejected_loss_mask") == "xx"
    # The attention mask still covers the full trajectory, observation included.
    assert "obs1" in _text("chosen", "chosen_attention_mask")
    assert "obs1" in _text("rejected", "rejected_attention_mask")
    assert "obs1" not in _text("chosen", "chosen_loss_mask")
    # Skipped groups keep a row (static shape) with all-zero loss masks.
    for row in (1, 2):
        assert not np.asarray(batch["chosen_loss_mask"][row]).any()
        assert not np.asarray(batch["rejected_loss_mask"][row]).any()


def _bias_for(batch, seed):
    import jax
    from jax import numpy as jnp

    rows = 2 * batch["prompt_input_ids"].shape[0]
    seq_len = batch["prompt_input_ids"].shape[1] + batch["chosen_input_ids"].shape[1]
    return jax.random.normal(jax.random.PRNGKey(seed), (rows, seq_len, VOCAB), dtype=jnp.float32)


def test_dpo_explicit_loss_mask_matches_numpy_and_observation_tokens_get_zero_gradient():
    import jax
    from easydel.trainers.direct_preference_optimization_trainer._fn import compute_dpo_losses
    from jax import numpy as jnp

    _, batch, _ = _make_flow_and_batch()
    bias = _bias_for(batch, 0)
    ref_chosen = jnp.asarray([-3.0, 0.0, 0.0], dtype=jnp.float32)
    ref_rejected = jnp.asarray([-2.5, 0.0, 0.0], dtype=jnp.float32)

    out = _forward(bias, batch)
    ref = _numpy_logps(np.asarray(bias), batch, use_loss_mask=True)
    np.testing.assert_allclose(np.asarray(out["chosen_logps"]), ref["chosen"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out["rejected_logps"]), ref["rejected"], rtol=1e-5, atol=1e-5)
    # Skipped groups (all-zero loss masks) score nothing.
    np.testing.assert_array_equal(np.asarray(out["chosen_logps"][1:]), [0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(out["rejected_logps"][1:]), [0.0, 0.0])

    def loss_fn(b, data):
        o = _forward(b, data)
        losses = compute_dpo_losses(
            o["chosen_logps"],
            o["rejected_logps"],
            ref_chosen,
            ref_rejected,
            beta=BETA,
            label_smoothing=0.0,
            loss_type="sigmoid",
            chosen_lengths=o["chosen_lengths"],
            rejected_lengths=o["rejected_lengths"],
        )
        return losses.mean(), losses

    (_, losses), grad = jax.value_and_grad(loss_fn, has_aux=True)(bias, batch)
    expected = _numpy_sigmoid_dpo(ref["chosen"], ref["rejected"], np.asarray(ref_chosen), np.asarray(ref_rejected))
    np.testing.assert_allclose(np.asarray(losses), expected, rtol=1e-5, atol=1e-6)

    # Gradient is nonzero exactly at the logits that score action tokens.
    scored = _scored_positions(batch, use_loss_mask=True)
    attended = _scored_positions(batch, use_loss_mask=False)
    observation = attended & ~scored
    assert observation[0].any() and observation[3].any(), "the pair must contain observation tokens"
    grad_norm = np.abs(np.asarray(grad)).sum(-1)
    np.testing.assert_array_equal(grad_norm[observation], 0.0)
    np.testing.assert_array_equal(grad_norm[~scored], 0.0)
    assert (grad_norm[scored] > 0).all()

    # Without loss masks, the same trajectory puts gradient on the observation tokens.
    unmasked = {k: v for k, v in batch.items() if not k.endswith("_loss_mask")}
    (_, _), grad_unmasked = jax.value_and_grad(loss_fn, has_aux=True)(bias, unmasked)
    unmasked_norm = np.abs(np.asarray(grad_unmasked)).sum(-1)
    assert (unmasked_norm[0][observation[0]] > 0).all()
    assert (unmasked_norm[3][observation[3]] > 0).all()


def test_dpo_loss_reduction_ignores_placeholder_pairs():
    import jax
    from easydel.trainers.direct_preference_optimization_trainer._fn import _reduce_pair_losses, compute_dpo_losses
    from jax import numpy as jnp

    _, batch, _ = _make_flow_and_batch()
    bias = _bias_for(batch, 2)
    ref_chosen = jnp.asarray([-3.0, 0.0, 0.0], dtype=jnp.float32)
    ref_rejected = jnp.asarray([-2.5, 0.0, 0.0], dtype=jnp.float32)

    def reduced_loss(b, data):
        o = _forward(b, data)
        losses = compute_dpo_losses(
            o["chosen_logps"],
            o["rejected_logps"],
            ref_chosen[: data["prompt_input_ids"].shape[0]],
            ref_rejected[: data["prompt_input_ids"].shape[0]],
            beta=BETA,
            label_smoothing=0.0,
            loss_type="sigmoid",
            chosen_lengths=o["chosen_lengths"],
            rejected_lengths=o["rejected_lengths"],
        )
        return _reduce_pair_losses(losses, o)

    out = _forward(bias, batch)
    np.testing.assert_array_equal(np.asarray(out["pair_valid"]), [1.0, 0.0, 0.0])

    # Only the one real pair enters the mean: no log(2) per placeholder row, no 1/3 gradient scaling.
    ref = _numpy_logps(np.asarray(bias), batch, use_loss_mask=True)
    expected = _numpy_sigmoid_dpo(ref["chosen"][:1], ref["rejected"][:1], -3.0, -2.5)[0]
    loss, grad = jax.value_and_grad(reduced_loss)(bias, batch)
    np.testing.assert_allclose(float(loss), expected, rtol=1e-5, atol=1e-6)

    # The gradient equals the gradient of the single real pair on its own.
    only = {k: v[:1] for k, v in batch.items()}
    rows = [0, 3]  # chosen row 0 and rejected row 0 in the concatenated [2N, L] layout
    _, grad_only = jax.value_and_grad(reduced_loss)(bias[jnp.asarray(rows)], only)
    np.testing.assert_allclose(np.asarray(grad)[rows], np.asarray(grad_only), rtol=1e-5, atol=1e-6)
    assert not np.asarray(grad)[[1, 2, 4, 5]].any()


def test_dpo_default_path_without_loss_masks_is_unchanged():
    from jax import numpy as jnp

    _, batch, _ = _make_flow_and_batch()
    unmasked = {k: v for k, v in batch.items() if not k.endswith("_loss_mask")}
    bias = _bias_for(unmasked, 1)

    out = _forward(bias, unmasked)
    ref = _numpy_logps(np.asarray(bias), unmasked, use_loss_mask=False)
    np.testing.assert_allclose(np.asarray(out["chosen_logps"]), ref["chosen"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out["rejected_logps"]), ref["rejected"], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(out["chosen_lengths"]), np.maximum(np.asarray(unmasked["chosen_attention_mask"]).sum(-1), 1)
    )

    # All-ones loss masks select exactly the attended tokens: bit-identical to omitting them.
    ones = {
        **unmasked,
        "chosen_loss_mask": jnp.ones_like(unmasked["chosen_attention_mask"]),
        "rejected_loss_mask": jnp.ones_like(unmasked["rejected_attention_mask"]),
    }
    out_ones = _forward(bias, ones)
    for key in out:
        np.testing.assert_array_equal(np.asarray(out[key]), np.asarray(out_ones[key]))
