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

"""GMPO, GDPO, REINFORCE++, Kimi K2.5 log-mask/Toggle and DAPO dynamic sampling on the GRPO stack."""

import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.trainers.base_trainer import GenerationResults
from easydel.trainers.gdpo_trainer import GDPOConfig, GDPOTrainer
from easydel.trainers.gmpo_trainer import GMPOConfig, GMPOTrainer
from easydel.trainers.group_relative_policy_optimization._fn import (
    _compute_grpo_policy_loss_terms,
    _compute_per_token_kl,
    grpo_step,
)
from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import (
    GRPOTrainer,
    _compute_rewards_and_advantages,
    _ScoredRollout,
)
from easydel.trainers.reinforce_plus_plus_trainer import ReinforcePlusPlusConfig, ReinforcePlusPlusTrainer
from easydel.trainers.training_utils import ScheduledStepCall, get_scheduled_loss_adapter
from easydel.utils import Registry
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec

VOCAB = 5


def _stub_logits(input_ids, bias, xp):
    """Logits of the toy policy: a learnable per-position bias plus a fixed input-dependent tilt."""
    tilt = 0.1 * input_ids[:, :, None].astype(xp.float32) * xp.arange(VOCAB, dtype=xp.float32)[None, None, :]
    return bias[None, : input_ids.shape[1], :] + tilt


class _Model:
    def __init__(self, bias, mesh):
        self.bias = bias
        self.mesh = mesh

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=_stub_logits(input_ids, self.bias, jnp))


class _State:
    def __init__(self, graphstate, mesh):
        self.mesh = mesh
        self.model = _Model(graphstate, mesh)
        self.graphstate = graphstate
        self.graphdef = None
        self.graphother = None

    def replace(self, **kwargs):
        return _State(kwargs.get("graphstate", self.graphstate), self.mesh)

    def merge(self, tree):
        return _Model(tree, self.mesh)


def _scheduled_loss_fn(state, batch, **kwargs):
    bound = inspect.signature(grpo_step).bind(
        state,
        batch,
        2,
        kwargs.pop("beta", 0.0),
        is_training=False,
        partition_spec=PartitionSpec(),
        **kwargs,
    )
    bound.apply_defaults()
    call = ScheduledStepCall(
        step_fn=grpo_step, state=state, batch=batch, args=(), kwargs={}, bound_arguments=bound.arguments, schedule=None
    )
    return get_scheduled_loss_adapter(grpo_step).make_loss(call)


def _reference_policy_logps(prompt_ids, completion_ids, bias, num_generations):
    input_ids = np.concatenate([np.repeat(prompt_ids, num_generations, 0), completion_ids], axis=1)
    logits = _stub_logits(input_ids, np.asarray(bias), np).astype(np.float64)
    prompt_len = prompt_ids.shape[1]
    logits = logits[:, prompt_len - 1 : -1]
    log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    return np.take_along_axis(log_probs, completion_ids[..., None], -1)[..., 0]


def _loss_batch():
    rng = np.random.default_rng(0)
    prompt_ids = np.array([[1], [3]], dtype=np.int32)
    completion_ids = np.array([[0, 2, 4], [1, 1, 3], [4, 0, 2], [3, 2, 1]], dtype=np.int32)
    completion_mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=np.float32)
    bias = rng.normal(size=(4, VOCAB)).astype(np.float32)
    policy = _reference_policy_logps(prompt_ids, completion_ids, bias, 2)
    old = policy + rng.uniform(-0.8, 0.8, size=policy.shape)
    ref = policy + rng.uniform(-0.5, 0.5, size=policy.shape)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.ones_like(jnp.asarray(prompt_ids)),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(completion_mask),
        "advantages": jnp.asarray([1.5, -0.7, 0.4, -1.2], dtype=jnp.float32),
        "old_per_token_logps": jnp.asarray(old, dtype=jnp.float32),
        "ref_per_token_logps": jnp.asarray(ref, dtype=jnp.float32),
    }
    return batch, jnp.asarray(bias), policy, old, ref, completion_mask


def _mesh():
    return Mesh(np.array(jax.devices()[:1]), ("dp",))


def _gmpo_reference(policy, old, advantages, mask, eps_low, eps_high):
    log_ratio = policy - old
    clipped = np.where(
        advantages[:, None] > 0,
        np.minimum(log_ratio, np.log(1 + eps_high)),
        np.maximum(log_ratio, np.log(1 - eps_low)),
    )
    clipped = np.maximum(clipped, np.log(1e-6))
    geo = np.exp((clipped * mask).sum(-1) / np.maximum(mask.sum(-1), 1.0))
    return -advantages * geo


def test_gmpo_policy_terms_match_log_space_geometric_mean_reference():
    per_token_logps = jnp.asarray([[0.3, -0.4, 0.05, 9.0], [-0.6, 0.2, 0.1, -9.0]], dtype=jnp.float32)
    old = jnp.zeros_like(per_token_logps)
    mask = jnp.asarray([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=jnp.float32)
    advantages = jnp.asarray([[2.0], [-1.0]], dtype=jnp.float32)

    loss, ratios = _compute_grpo_policy_loss_terms(
        per_token_logps=per_token_logps,
        old_per_token_logps=old,
        advantages=advantages,
        completion_mask=mask,
        loss_type="gmpo",
        epsilon=0.4,
        epsilon_high=0.4,
        delta=None,
        importance_sampling_level="token",
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
    )

    expected = _gmpo_reference(
        np.asarray(per_token_logps, np.float64), 0.0, np.array([2.0, -1.0]), np.asarray(mask), 0.4, 0.4
    )
    assert loss.shape == per_token_logps.shape
    assert np.isfinite(np.asarray(loss)).all()
    np.testing.assert_allclose(np.asarray(loss[:, 0]), expected, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(loss), np.broadcast_to(expected[:, None], loss.shape), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(ratios), np.exp(np.asarray(per_token_logps)), rtol=1e-5)


@pytest.mark.parametrize("chunk_size", [None, 1, 3])
def test_gmpo_grpo_step_matches_reference_for_chunked_and_unchunked_paths(chunk_size):
    batch, bias, policy, old, _, mask = _loss_batch()
    state = _State(bias, _mesh())
    loss_fn = _scheduled_loss_fn(
        state, batch, loss_type="gmpo", epsilon=0.3, epsilon_high=0.25, completion_chunk_size=chunk_size
    )
    loss = loss_fn(bias, batch)

    per_sequence = _gmpo_reference(policy, old, np.asarray(batch["advantages"], np.float64), mask, 0.3, 0.25)
    np.testing.assert_allclose(np.asarray(loss), per_sequence.mean(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("advantage", "old_shift"), [(1.0, -1.0), (-1.0, 1.0)])
def test_gmpo_sign_aware_clip_blocks_gradient_of_out_of_trust_region_tokens(advantage, old_shift):
    bias = jnp.zeros((3, VOCAB), dtype=jnp.float32)
    state = _State(bias, _mesh())
    completion_ids = np.array([[1, 2]], dtype=np.int32)
    prompt_ids = np.array([[0]], dtype=np.int32)
    policy = _reference_policy_logps(prompt_ids, completion_ids, bias, 1)
    # Token 0 sits outside the trust region in the direction the advantage pushes; token 1 is on-policy.
    old = policy + np.array([[old_shift, 0.0]])
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.ones((1, 1), dtype=jnp.int32),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "advantages": jnp.asarray([advantage], dtype=jnp.float32),
        "old_per_token_logps": jnp.asarray(old, dtype=jnp.float32),
    }
    bound = inspect.signature(grpo_step).bind(
        state,
        batch,
        1,
        0.0,
        is_training=False,
        partition_spec=PartitionSpec(),
        loss_type="gmpo",
        epsilon=0.4,
        epsilon_high=0.4,
    )
    bound.apply_defaults()
    call = ScheduledStepCall(
        step_fn=grpo_step, state=state, batch=batch, args=(), kwargs={}, bound_arguments=bound.arguments, schedule=None
    )
    loss, grads = jax.value_and_grad(get_scheduled_loss_adapter(grpo_step).make_loss(call))(bias, batch)

    clipped_log = np.log(1.4) if advantage > 0 else np.log(0.6)
    np.testing.assert_allclose(np.asarray(loss), -advantage * np.exp(clipped_log / 2), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(grads[0]), np.zeros(VOCAB), atol=1e-7)
    assert float(jnp.max(jnp.abs(grads[1]))) > 1e-3


@pytest.mark.parametrize(
    ("estimator", "formula"),
    [
        ("k1", lambda lr: lr),
        ("k2", lambda lr: 0.5 * lr**2),
        ("k3", lambda lr: np.exp(-lr) - 1 + lr),
        ("l2", lambda lr: lr**2),
    ],
)
def test_kl_estimators_match_closed_forms(estimator, formula):
    policy = np.array([[-1.0, -2.5, -0.3]], dtype=np.float32)
    ref = np.array([[-1.4, -2.0, -0.3]], dtype=np.float32)
    kl = _compute_per_token_kl(jnp.asarray(policy), jnp.asarray(ref), estimator)
    # float32 exp on TPU: measured relative error ~1.4e-5.
    np.testing.assert_allclose(np.asarray(kl), formula(policy.astype(np.float64) - ref), rtol=2e-5, atol=1e-7)
    with pytest.raises(ValueError, match="kl_estimator"):
        _compute_per_token_kl(jnp.asarray(policy), jnp.asarray(ref), "k9")


@pytest.mark.parametrize("estimator", ["k1", "k2", "k3", "l2"])
@pytest.mark.parametrize("chunk_size", [None, 1])
def test_grpo_step_kl_penalty_uses_selected_estimator(estimator, chunk_size):
    batch, bias, policy, _, ref, mask = _loss_batch()
    state = _State(bias, _mesh())
    beta = 0.3
    common = {"loss_type": "grpo", "completion_chunk_size": chunk_size, "kl_estimator": estimator}
    with_kl = _scheduled_loss_fn(state, batch, beta=beta, **common)(bias, batch)
    without_kl = _scheduled_loss_fn(state, batch, beta=0.0, **common)(bias, batch)

    log_ratio = policy - ref
    per_token = {
        "k1": log_ratio,
        "k2": 0.5 * log_ratio**2,
        "k3": np.exp(-log_ratio) - 1 + log_ratio,
        "l2": log_ratio**2,
    }[estimator]
    per_sequence = (per_token * mask).sum(-1) / mask.sum(-1)
    np.testing.assert_allclose(np.asarray(with_kl - without_kl), beta * per_sequence.mean(), rtol=1e-4, atol=1e-6)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("is_train", [True, False])
def test_static_arg_tuples_bind_to_grpo_step_signature(is_train):
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(
        kl_estimator="k2",
        loss_type="gmpo",
        epsilon=0.3,
        off_policy_k3_threshold=0.5,
        off_policy_min_acceptance=0.75,
        startup_behavior_k3_max=0.02,
        startup_behavior_tail_fraction_max=0.05,
    )
    trainer.scheduler = None
    trainer.loss_type = trainer.arguments.loss_type
    trainer.epsilon = trainer.arguments.epsilon
    trainer.epsilon_high = trainer.arguments.epsilon_high
    trainer.delta = trainer.arguments.delta
    trainer.importance_sampling_level = "token"
    trainer.top_entropy_quantile = 1.0
    trainer.kl_estimator = "k2"
    static_args = trainer._grpo_step_static_args(num_generations=3, is_train=is_train, straight_through_emulator=None)

    bound = inspect.signature(grpo_step).bind(object(), {}, *static_args).arguments
    assert bound["num_generations"] == 3
    assert bound["is_training"] is is_train
    assert bound["loss_type"] == "gmpo"
    assert bound["epsilon"] == 0.3
    assert bound["kl_estimator"] == "k2"
    assert bound["off_policy_k3_threshold"] == 0.5
    assert bound["off_policy_min_acceptance"] == 0.75
    assert bound["startup_behavior_k3_max"] == 0.02
    assert bound["startup_behavior_tail_fraction_max"] == 0.05
    # Every parameter after (state, batch) is bound positionally: nothing silently falls back to a default.
    assert len(bound) == len(inspect.signature(grpo_step).parameters)
    assert get_scheduled_loss_adapter(grpo_step) is not None


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_gmpo_token_is_mask_blocks_gradient_inside_geometric_mean(chunk_size):
    bias = jnp.zeros((3, VOCAB), dtype=jnp.float32)
    state = _State(bias, _mesh())
    prompt_ids = np.array([[0]], dtype=np.int32)
    completion_ids = np.array([[1, 2], [3, 4]], dtype=np.int32)
    advantages = np.array([1.0, 2.0], dtype=np.float32)
    # Token 0 of every row is masked by a token-level rollout IS mode; token 1 is truncated to weight 0.5.
    is_ratio = np.array([[0.0, 0.5], [0.0, 0.5]], dtype=np.float32)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.ones((1, 1), dtype=jnp.int32),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.ones((2, 2), dtype=jnp.float32),
        "advantages": jnp.asarray(advantages),
        "importance_sampling_ratio": jnp.asarray(is_ratio),
    }
    loss_fn = _scheduled_loss_fn(
        state, batch, loss_type="gmpo", epsilon=0.4, epsilon_high=0.4, completion_chunk_size=chunk_size
    )
    loss, grads = jax.value_and_grad(loss_fn)(bias, batch)

    # On-policy (old == current): only token 1 survives the mask, its log ratio is 0 and its
    # constant log-weight log(0.5) enters the geometric mean, so each sequence ratio is 0.5.
    np.testing.assert_allclose(np.asarray(loss), np.mean(-advantages * 0.5), rtol=1e-5)
    # Masked tokens (position 0) receive exactly zero gradient through the sequence ratio.
    np.testing.assert_array_equal(np.asarray(grads[0]), np.zeros(VOCAB, dtype=np.float32))
    assert float(jnp.max(jnp.abs(grads[1]))) > 1e-3


def test_reinforce_plus_plus_batch_advantage_excludes_between_group_variance():
    rewards_per_func = jnp.asarray([[10.0], [10.0], [10.0], [12.0], [0.0], [0.0], [0.0], [2.0]], dtype=jnp.float32)
    common = {
        "rewards_per_func": rewards_per_func,
        "reward_weights": jnp.ones((1,), dtype=jnp.float32),
        "generation_factor": 4,
        "multi_objective_aggregation": "sum_then_normalize",
        "arguments": ReinforcePlusPlusConfig(),
    }
    _, advantages, std, _ = _compute_rewards_and_advantages(scale_rewards="batch_advantage", **common)

    # Group means are 10.5 and 0.5, so A = [-.5, -.5, -.5, 1.5] in both groups; mean(A) = 0, std(A) = sqrt(0.75).
    centred = np.array([-0.5, -0.5, -0.5, 1.5] * 2)
    np.testing.assert_allclose(np.asarray(advantages), centred / (np.sqrt(0.75) + 1e-4), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(std), np.full(8, np.sqrt(0.75)), rtol=1e-5)

    # The raw-reward batch std (~5.1) is dominated by between-group variance and shrinks the advantages.
    _, batch_advantages, _, _ = _compute_rewards_and_advantages(scale_rewards="batch", **common)
    raw = np.array([10, 10, 10, 12, 0, 0, 0, 2], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(batch_advantages), centred / (raw.std() + 1e-4), rtol=1e-5)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        # Ratios 4, 1/4, 1.5 (masked out), and per-row log-ratio sums log(1)=0 / log(3).
        ("token_log_mask", [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]),
        ("sequence_log_mask", [[1.0], [0.0]]),
    ],
)
def test_kimi_log_mask_modes_drop_large_absolute_log_ratios(mode, expected):
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = SimpleNamespace(esurge_importance_sampling_mode=mode, esurge_importance_sampling_cap=2.0)
    policy = jnp.log(jnp.asarray([[4.0, 0.25, 1.5], [1.5, 2.0, 7.0]], dtype=jnp.float32))
    mask = jnp.asarray([[1, 1, 0], [1, 1, 0]], dtype=jnp.float32)
    ratio = trainer._esurge_importance_sampling_ratio(policy, jnp.zeros_like(policy), mask)
    np.testing.assert_array_equal(np.asarray(ratio), expected)


@pytest.mark.cpu_ok
def test_grpo_config_validates_new_survey_knobs():
    cfg = GRPOConfig(
        loss_type="GMPO",
        kl_estimator="K2",
        toggle_budget_interval=4,
        toggle_max_completion_length=64,
        dynamic_sampling=True,
        esurge_importance_sampling_mode="token_log_mask",
    )
    assert cfg.loss_type == "gmpo"
    assert cfg.kl_estimator == "k2"
    assert cfg.toggle_max_completion_length == 64
    assert cfg.dynamic_sampling_max_attempts == 3
    assert GRPOConfig().kl_estimator == "k3"

    with pytest.raises(ValueError, match="importance_sampling_level"):
        GRPOConfig(loss_type="gmpo", importance_sampling_level="sequence")
    with pytest.raises(ValueError, match="epsilon"):
        GRPOConfig(loss_type="gmpo", epsilon=1.0)
    with pytest.raises(ValueError, match="kl_estimator"):
        GRPOConfig(kl_estimator="k4")
    with pytest.raises(ValueError, match="toggle_max_completion_length"):
        GRPOConfig(toggle_budget_interval=2)
    with pytest.raises(ValueError, match="toggle_max_completion_length"):
        GRPOConfig(max_completion_length=128, toggle_budget_interval=2, toggle_max_completion_length=256)
    with pytest.raises(ValueError, match="toggle_budget_interval"):
        GRPOConfig(toggle_budget_interval=-1)
    with pytest.raises(ValueError, match="dynamic_sampling_max_attempts"):
        GRPOConfig(dynamic_sampling_max_attempts=0)
    with pytest.raises(ValueError, match="num_generations"):
        GRPOConfig(dynamic_sampling=True, num_generations=1)
    with pytest.raises(ValueError, match="log-mask"):
        GRPOConfig(esurge_importance_sampling_mode="sequence_log_mask", esurge_importance_sampling_cap=1.0)


@pytest.mark.cpu_ok
def test_decouple_reward_normalization_selects_normalize_then_sum():
    assert GRPOConfig().decouple_reward_normalization is False
    cfg = GRPOConfig(decouple_reward_normalization=True)
    assert cfg.multi_objective_aggregation == "normalize_then_sum"
    assert GRPOConfig(multi_objective_aggregation="normalize_then_sum").decouple_reward_normalization is True


@pytest.mark.cpu_ok
def test_survey_trainer_configs_defaults_and_registry():
    gmpo = GMPOConfig()
    assert (gmpo.loss_type, gmpo.epsilon, gmpo.epsilon_high) == ("gmpo", 0.4, 0.4)
    gdpo = GDPOConfig()
    assert gdpo.decouple_reward_normalization is True
    assert gdpo.multi_objective_aggregation == "normalize_then_sum"
    rpp = ReinforcePlusPlusConfig()
    assert (rpp.scale_rewards, rpp.kl_estimator) == ("batch_advantage", "k2")

    for name, config_cls, trainer_cls in (
        ("gmpo", GMPOConfig, GMPOTrainer),
        ("gdpo", GDPOConfig, GDPOTrainer),
        ("reinforce_plus_plus", ReinforcePlusPlusConfig, ReinforcePlusPlusTrainer),
    ):
        assert Registry.get("trainer-arguments", name) is config_cls
        assert Registry.get("trainer", name) is trainer_cls
        assert issubclass(trainer_cls, GRPOTrainer)
        with pytest.raises(TypeError, match=config_cls.__name__):
            trainer_cls(arguments=GRPOConfig(), model=None, reward_funcs=[])


@pytest.mark.cpu_ok
def test_toggle_alternates_generation_budget_by_step_period():
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(
        num_generations=2,
        max_completion_length=8,
        toggle_budget_interval=2,
        toggle_max_completion_length=3,
    )
    budgets = [trainer._generation_config_overrides_for_phase(True, step=s).get("max_new_tokens") for s in range(6)]
    assert budgets == [None, None, 3, 3, None, None]
    assert trainer._generation_config_overrides_for_phase(False, step=2) is None
    trainer.arguments = GRPOConfig(num_generations=2)
    assert trainer._generation_config_overrides_for_phase(True, step=3) == {"num_return_sequences": 2}


def test_toggle_budget_rollout_is_padded_to_full_width_through_preprocessing():
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(
        num_generations=2,
        max_prompt_length=1,
        max_completion_length=4,
        toggle_budget_interval=1,
        toggle_max_completion_length=2,
        beta=0.0,
        esurge_importance_sampling_correction=False,
        scale_rewards="none",
    )
    trainer.steps_per_generation = 1
    trainer.num_iterations = 1
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_remaining = 0
    trainer.ref_state = None
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.scale_rewards = "none"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.reward_weights = jnp.ones((1,))
    trainer.reward_funcs = [lambda **kwargs: [1.0, 2.0, 5.0, 9.0]]
    trainer.reward_processing_classes = [None]
    trainer.reward_func_names = ["reward"]
    trainer._pad_token_id = 0
    trainer._apply_user_data_collator = lambda batch: batch
    trainer._extract_reward_batch_sidechannels = lambda batch: {}
    trainer._purify_batch = lambda batch: batch
    trainer._all_gather = lambda value: value
    trainer._make_attn_mask = lambda ids: (ids != 0).astype(jnp.int32)
    trainer._tool_call_loop_enabled = lambda: False
    trainer._run_environment_feedback = lambda **kwargs: None
    trainer._build_reward_call_kwargs = lambda func, **kwargs: kwargs
    trainer._maybe_log_grpo_completions = lambda **kwargs: None
    trainer._log_training_generations_to_wandb = lambda **kwargs: None
    prompt_ids = jnp.asarray([[1], [2]])
    ids = jnp.asarray([[5, 6], [7, 0], [8, 9], [3, 0]])
    texts = [f"c{i}" for i in range(4)]
    seen_overrides = []

    def rollout(**kwargs):
        seen_overrides.append(kwargs["config_overrides"])
        return GenerationResults(
            generation_results=texts,
            prompt_ids=prompt_ids,
            prompt_mask=jnp.ones_like(prompt_ids),
            sequences=jnp.concatenate([prompt_ids.repeat(2, 0), ids], axis=1),
            completion_ids=ids,
            completion_mask=(ids != 0).astype(jnp.int32),
            decoded_prompts=["p0", "p1"],
            completion_prompts=["p0", "p0", "p1", "p1"],
            text=texts,
            raw_text=texts,
            sampling_per_token_logps=jnp.full(ids.shape, -0.5),
        )

    trainer.rollout = rollout
    state = SimpleNamespace(model=lambda **kwargs: None, step=jnp.asarray(1))
    batch = {"input_ids": prompt_ids, "attention_mask": jnp.ones_like(prompt_ids)}

    model_batch, metrics = trainer._preprocess_batch_input(state, batch, True)

    assert seen_overrides == [{"num_return_sequences": 2, "max_new_tokens": 2}]
    assert metrics["toggle/budget_constrained"] == 1
    np.testing.assert_array_equal(
        np.asarray(model_batch["completion_ids"]), [[5, 6, 0, 0], [7, 0, 0, 0], [8, 9, 0, 0], [3, 0, 0, 0]]
    )
    np.testing.assert_array_equal(
        np.asarray(model_batch["completion_mask"]), [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]]
    )
    assert int(model_batch["num_items_in_batch"]) == 6


def _scored(attempt, rewards):
    rewards = np.asarray(rewards, dtype=np.float32)
    rows = rewards.shape[0]
    return _ScoredRollout(
        model_batch={
            "prompt_ids": jnp.ones((rows // 2, 1), dtype=jnp.int32),
            "prompt_mask": jnp.ones((rows // 2, 1), dtype=jnp.int32),
            "completion_ids": jnp.full((rows, 2), attempt, dtype=jnp.int32),
            "completion_mask": jnp.ones((rows, 2), dtype=jnp.int32),
            "completion_attention_mask": jnp.ones((rows, 2), dtype=jnp.int32),
            "ref_per_token_logps": jnp.full((rows, 2), -float(attempt)),
            "advantages": jnp.zeros((rows,), dtype=jnp.float32),
            "num_items_in_batch": jnp.asarray(rows * 2),
        },
        metrics={"generation_time": 1.0, "reward_mean": 0.0},
        rewards_per_func=jnp.asarray(rewards[:, None]),
        reward_weights=jnp.ones((1,), dtype=jnp.float32),
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_dapo_dynamic_sampling_resamples_then_filters_zero_variance_groups(enabled):
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(num_generations=2, dynamic_sampling=enabled, dynamic_sampling_max_attempts=3)
    trainer.scale_rewards = "group"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.steps_per_generation = 1
    trainer.num_iterations = 1
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_remaining = 0
    # Groups: [1,1] constant, [0,1] informative, [2,2] constant. Attempt 2 fixes group 0; group 2 never varies.
    attempts = iter(
        [
            _scored(1, [1, 1, 0, 1, 2, 2]),
            _scored(2, [0, 1, 4, 4, 3, 3]),
            _scored(3, [7, 7, 0, 5, 5, 5]),
        ]
    )
    calls = []

    def generate(state, batch, is_train):
        calls.append(is_train)
        return next(attempts)

    trainer._generate_and_score_batch = generate
    model_batch, metrics = trainer._preprocess_batch_input(None, {}, True)

    if not enabled:
        assert calls == [True]
        assert "dynamic_sampling/attempts" not in metrics
        return
    assert calls == [True, True, True]
    np.testing.assert_array_equal(np.asarray(model_batch["completion_ids"][:, 0]), [2, 2, 1, 1, 1, 1])
    np.testing.assert_array_equal(np.asarray(model_batch["ref_per_token_logps"][:, 0]), [-2, -2, -1, -1, -1, -1])
    np.testing.assert_array_equal(np.asarray(model_batch["completion_mask"]).sum(-1), [2, 2, 2, 2, 0, 0])
    assert int(model_batch["num_items_in_batch"]) == 8
    scale = 0.5 / (0.5 + 1e-4)
    np.testing.assert_allclose(np.asarray(model_batch["advantages"]), [-scale, scale, -scale, scale, 0, 0], rtol=1e-5)
    assert metrics["dynamic_sampling/attempts"] == 3
    assert metrics["dynamic_sampling/initial_zero_std_group_frac"] == pytest.approx(2 / 3)
    assert metrics["dynamic_sampling/resampled_group_frac"] == pytest.approx(1 / 3)
    assert metrics["dynamic_sampling/filtered_group_frac"] == pytest.approx(1 / 3)
    assert metrics["generation_time"] == pytest.approx(3.0)


def test_dapo_dynamic_sampling_stops_once_every_group_is_informative():
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(num_generations=2, dynamic_sampling=True, dynamic_sampling_max_attempts=5)
    trainer.scale_rewards = "group"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.steps_per_generation = 1
    trainer.num_iterations = 1
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_remaining = 0
    attempts = iter([_scored(1, [1, 1, 0, 1]), _scored(2, [0, 3, 9, 9])])
    trainer._generate_and_score_batch = lambda state, batch, is_train: next(attempts)

    model_batch, metrics = trainer._preprocess_batch_input(None, {}, True)

    assert metrics["dynamic_sampling/attempts"] == 2
    assert metrics["dynamic_sampling/filtered_group_frac"] == 0.0
    np.testing.assert_array_equal(np.asarray(model_batch["completion_mask"]).sum(-1), [2, 2, 2, 2])
    np.testing.assert_array_equal(np.asarray(model_batch["completion_ids"][:, 0]), [2, 2, 1, 1])


def test_dynamic_sampling_resampled_group_does_not_inherit_discard_weight():
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(
        num_generations=2, dynamic_sampling=True, dynamic_sampling_max_attempts=3, discard_unscorable_groups=True
    )
    trainer.scale_rewards = "group"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.steps_per_generation = 1
    trainer.num_iterations = 1
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_remaining = 0
    trainer.reward_funcs = []
    # First rollout: group 0 is unscorable (discarded, weight 0, mask 0); the candidate discards nothing.
    first = _scored(1, [np.nan, 1, 0, 1])
    first.model_batch["difficulty_weights"] = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    first.model_batch["completion_mask"] = jnp.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=jnp.int32)
    attempts = iter([first, _scored(2, [0, 1, 5, 5])])
    trainer._generate_and_score_batch = lambda state, batch, is_train: next(attempts)

    model_batch, metrics = trainer._preprocess_batch_input(None, {}, True)

    assert metrics["dynamic_sampling/resampled_group_frac"] == pytest.approx(0.5)
    np.testing.assert_array_equal(np.asarray(model_batch["completion_ids"][:, 0]), [2, 2, 1, 1])
    np.testing.assert_array_equal(np.asarray(model_batch["completion_mask"]).sum(-1), [2, 2, 2, 2])
    # The resampled, scorable group trains with weight 1 instead of the discarded group's 0.
    np.testing.assert_array_equal(np.asarray(model_batch["difficulty_weights"]), [1.0, 1.0, 1.0, 1.0])
    assert float(jnp.abs(model_batch["advantages"][:2]).min()) > 0.5


@pytest.mark.cpu_ok
def test_elarge_reinforce_plus_plus_preset_uses_batch_advantage_scaling():
    from easydel.infra.elarge.types.training import get_trainer_defaults

    assert get_trainer_defaults("reinforce_plus_plus")["scale_rewards"] == ReinforcePlusPlusConfig().scale_rewards
    assert get_trainer_defaults("reinforce_plus_plus")["scale_rewards"] == "batch_advantage"


def test_gmpo_row_without_kept_tokens_has_zero_loss():
    per_token_logps = jnp.asarray([[0.3, -0.4], [-0.6, 0.2]], dtype=jnp.float32)
    loss, _ = _compute_grpo_policy_loss_terms(
        per_token_logps=per_token_logps,
        old_per_token_logps=jnp.zeros_like(per_token_logps),
        advantages=jnp.asarray([[2.0], [-1.0]], dtype=jnp.float32),
        completion_mask=jnp.ones((2, 2), dtype=jnp.float32),
        loss_type="gmpo",
        epsilon=0.4,
        epsilon_high=0.4,
        delta=None,
        importance_sampling_level="token",
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
        # Row 0: every token IS-masked; row 1: one token kept with weight 1.
        importance_sampling_ratio=jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(np.asarray(loss[0]), [0.0, 0.0])
    # Row 1: A = -1, the kept token's log ratio -0.6 is clipped up to log(0.6), so the loss is -A * 0.6.
    np.testing.assert_allclose(np.asarray(loss[1]), [0.6, 0.6], rtol=1e-5)
