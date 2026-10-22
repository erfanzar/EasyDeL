"""Accelerator-only GRPO log-probability numerics regressions."""

import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.trainers.base_trainer import GenerationResults
from easydel.trainers.group_filtered_policy_optimization import GFPOTrainer
from easydel.trainers.group_relative_policy_optimization._fn import get_per_token_logps_and_selected_logps, grpo_step
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from easydel.trainers.kto_trainer._fn import evaluation_step
from easydel.trainers.metrics import MetricsHistogram
from easydel.trainers.training_utils import ScheduledStepCall, get_scheduled_loss_adapter
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_histogram_counts_large_constant_low_precision_tensor(dtype):
    histogram = MetricsHistogram.from_array(jnp.zeros((70001,), dtype=dtype))
    counts, edges = histogram.numpy_histogram()
    expected_counts, expected_edges = np.histogram(np.zeros(70001, dtype=np.float32), bins=64)
    np.testing.assert_array_equal(np.asarray(counts), expected_counts)
    np.testing.assert_allclose(np.asarray(edges), expected_edges)
    assert counts.dtype == jnp.int32


@pytest.mark.parametrize("chunk_size", [None, 2, 3])
def test_selected_logps_match_independent_full_vocab_reference(chunk_size):
    logits = jnp.arange(21, dtype=jnp.float32).reshape(1, 3, 7) / 5

    def model(**kwargs):
        return SimpleNamespace(logits=logits)

    ids = jnp.array([[0, 3, 1]])
    selected = jnp.array([[[0, 6], [1, 4]]])
    token_logps, selected_logps = get_per_token_logps_and_selected_logps(
        model, ids, jnp.ones_like(ids), 1, selected, logprob_vocab_chunk_size=chunk_size
    )
    reference = np.asarray(logits)[:, :2]
    reference = reference - np.log(np.exp(reference).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(
        np.asarray(token_logps), np.take_along_axis(reference, np.array([[[3], [1]]]), -1)[..., 0], atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(selected_logps), np.take_along_axis(reference, np.asarray(selected), -1), atol=2e-6
    )


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 3])
@pytest.mark.parametrize("target_logit", [-1e8, -jnp.inf])
def test_selected_logps_extreme_or_masked_target_preserve_normalizer(chunk_size, target_logit):
    logits = jnp.array([[[target_logit, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=jnp.float32)
    ids = jnp.array([[1, 0]])
    selected = jnp.array([[[1, 2]]])
    token_logps, selected_logps = get_per_token_logps_and_selected_logps(
        lambda **kwargs: SimpleNamespace(logits=logits),
        ids,
        jnp.ones_like(ids),
        1,
        selected,
        logprob_vocab_chunk_size=chunk_size,
    )
    # Independent host log-softmax: the masked target must not contaminate
    # finite selected scores, and a rounded -1e8 target must not erase log(2).
    reference_logits = np.array([target_logit, 0.0, 0.0], dtype=np.float64)
    reference = reference_logits - np.logaddexp.reduce(reference_logits)
    np.testing.assert_allclose(np.asarray(token_logps), [[reference[0]]], rtol=1e-7)
    np.testing.assert_allclose(np.asarray(selected_logps), [[[reference[1], reference[2]]]], atol=1e-7)
    assert np.isfinite(np.asarray(selected_logps)).all()


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("token_truncate", [[2.0, 1.0], [2.0, 2.0]]),
        ("token_mask", [[0.0, 1.0], [2.0, 2.0]]),
        ("sequence_truncate", [[2.0], [2.0]]),
        ("sequence_mask", [[0.0], [0.0]]),
    ],
)
def test_rollout_correction_modes(mode, expected):
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = SimpleNamespace(esurge_importance_sampling_mode=mode, esurge_importance_sampling_cap=2.0)
    ratio = trainer._esurge_importance_sampling_ratio(
        jnp.log(jnp.array([[4.0, 100.0], [2.0, 2.0]])), jnp.zeros((2, 2)), jnp.array([[1, 0], [1, 1]])
    )
    np.testing.assert_allclose(np.asarray(ratio), expected, atol=1e-6)


def test_scheduled_dppo_loss_matches_uniform_policy_reference():
    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))

    class Model:
        def __init__(self):
            self.mesh = mesh

        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=jnp.zeros((*input_ids.shape, 4)))

    class State:
        model = Model()
        graphstate = None
        graphdef = None
        graphother = None

        def replace(self, **kwargs):
            return self

        def merge(self, tree):
            return self.model

    state = State()
    batch = {
        "prompt_ids": jnp.array([[0]]),
        "prompt_mask": jnp.ones((1, 1)),
        "completion_ids": jnp.array([[1, 2]]),
        "completion_mask": jnp.ones((1, 2)),
        "advantages": jnp.ones((1,)),
        "sampling_per_token_logps": jnp.full((1, 2), -jnp.log(4.0)),
    }
    bound = inspect.signature(grpo_step).bind(
        state,
        batch,
        1,
        0.0,
        is_training=False,
        partition_spec=PartitionSpec(),
        loss_type="dppo",
        dppo_divergence_type="binary_tv",
        dppo_clip_ratio_c=3.0,
    )
    bound.apply_defaults()
    call = ScheduledStepCall(
        step_fn=grpo_step, state=state, batch=batch, args=(), kwargs={}, bound_arguments=bound.arguments, schedule=None
    )
    loss = get_scheduled_loss_adapter(grpo_step).make_loss(call)({}, batch)
    np.testing.assert_allclose(np.asarray(loss), np.log(4.0), rtol=1e-5)


def test_kto_cached_dataset_kl_does_not_change_runtime_pairing():
    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    state = SimpleNamespace(model=SimpleNamespace(mesh=mesh, scale=2.0))
    reference_state = SimpleNamespace(model=SimpleNamespace(mesh=mesh, scale=1.0))
    batch = {
        "prompt_input_ids": jnp.array([[1], [3]]),
        "prompt_attention_mask": jnp.ones((2, 1), dtype=jnp.int32),
        "completion_input_ids": jnp.array([[2], [5]]),
        "completion_attention_mask": jnp.ones((2, 1), dtype=jnp.int32),
        "label": jnp.array([True, False]),
    }

    def forward(model, batch):
        return {
            "completion_logps": model.scale * (batch["prompt_input_ids"][:, 0] * batch["completion_input_ids"][:, 0])
        }

    kwargs = dict(
        reference_state=reference_state,
        forward_fn=forward,
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        loss_type="kto",
        calculate_kl=True,
        aux_loss_coef=0.0,
    )
    actual = evaluation_step(state, {**batch, "reference_KL_logps": jnp.array([-1000.0, -2000.0])}, **kwargs)
    expected = evaluation_step(state, batch, **kwargs)
    np.testing.assert_allclose(np.asarray(actual.loss), np.asarray(expected.loss))
    np.testing.assert_allclose(np.asarray(actual.other_metrics["kl"]), np.asarray(expected.other_metrics["kl"]))


@pytest.mark.parametrize("correction_enabled", [False, True])
def test_gfpo_shared_rollout_preserves_rloo_reuse_and_log_alignment(correction_enabled):
    trainer = object.__new__(GFPOTrainer)
    trainer.arguments = SimpleNamespace(
        num_remains_in_group=2,
        num_generations=4,
        num_generations_eval=2,
        mask_truncated_completions=False,
        max_prompt_length=1,
        max_length=3,
        esurge_importance_sampling_correction=correction_enabled,
        esurge_importance_sampling_mode="token_truncate",
        esurge_importance_sampling_cap=2.0,
        logprob_vocab_chunk_size=2,
        advantage_estimator="leave_one_out",
        reward_clip_range=None,
    )
    trainer.group_filter_func = lambda ids, rewards, mask: ids[:, 0].astype(jnp.float32)
    trainer.steps_per_generation = 2
    trainer.num_iterations = 1
    trainer.ref_state = None
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.scale_rewards = "none"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.reward_weights = jnp.ones((1,))
    trainer.reward_funcs = [lambda **kwargs: [1.0, 2.0, 5.0, 9.0, 2.0, 3.0, 7.0, 10.0]]
    trainer.reward_processing_classes = [None]
    trainer.reward_func_names = ["reward"]
    trainer._pad_token_id = 0
    trainer._apply_user_data_collator = lambda batch: batch
    trainer._extract_reward_batch_sidechannels = lambda batch: {}
    trainer._purify_batch = lambda batch: batch
    trainer._all_gather = lambda value: value
    trainer._make_attn_mask = lambda ids: jnp.ones_like(ids)
    trainer._tool_call_loop_enabled = lambda: False
    trainer._run_environment_feedback = lambda **kwargs: None
    trainer._build_reward_call_kwargs = lambda func, **kwargs: kwargs
    trainer.compute_state_logps = lambda *args, **kwargs: jnp.zeros((8, 2))
    trainer._maybe_log_grpo_completions = lambda **kwargs: None
    logged = []
    trainer._log_training_generations_to_wandb = lambda **kwargs: logged.append(kwargs)
    ids = jnp.array([[11, 1], [12, 1], [13, 1], [14, 1], [21, 1], [22, 1], [23, 1], [24, 1]])
    prompts = ["first"] * 4 + ["second"] * 4
    texts = [f"answer-{i}" for i in range(8)]
    prompt_ids = jnp.array([[1], [2]])
    results = GenerationResults(
        generation_results=texts,
        prompt_ids=prompt_ids,
        prompt_mask=jnp.ones_like(prompt_ids),
        sequences=jnp.concatenate([prompt_ids.repeat(4, 0), ids], axis=1),
        completion_ids=ids,
        completion_mask=jnp.ones_like(ids),
        decoded_prompts=["first", "second"],
        completion_prompts=prompts,
        text=texts,
        raw_text=texts,
        sampling_per_token_logps=jnp.full(ids.shape, -jnp.log(4.0)),
    )
    trainer.rollout = lambda **kwargs: results
    state = SimpleNamespace(model=lambda **kwargs: None, step=jnp.array(0))
    batch = {"input_ids": prompt_ids, "attention_mask": jnp.ones_like(prompt_ids)}
    model_batch, metrics = trainer._preprocess_batch_input(state, batch, True)
    np.testing.assert_array_equal(np.asarray(model_batch["completion_ids"][:, 0]), [14, 13, 24, 23])
    np.testing.assert_allclose(np.asarray(model_batch["advantages"]), [4.0, -4.0, 3.0, -3.0])
    np.testing.assert_array_equal(np.asarray(model_batch["completion_attention_mask"]), np.ones((4, 2)))
    assert list(zip(logged[0]["prompts"], logged[0]["completions"], strict=True)) == list(
        zip(prompts, texts, strict=True)
    )
    reused_batch, reused_metrics = trainer._preprocess_batch_input(state, {}, True)
    np.testing.assert_array_equal(np.asarray(reused_batch["completion_ids"]), np.asarray(model_batch["completion_ids"]))
    assert metrics["generation_reused"] == 0
    assert reused_metrics["generation_reused"] == 1
    if correction_enabled:
        np.testing.assert_allclose(np.asarray(model_batch["importance_sampling_ratio"]), np.full((4, 2), 2.0))
        np.testing.assert_array_equal(
            np.asarray(reused_batch["importance_sampling_ratio"]), np.asarray(model_batch["importance_sampling_ratio"])
        )
    else:
        assert "importance_sampling_ratio" not in model_batch

    # Consume an actual batch from GRPO's shared preprocessing path, including
    # completion_attention_mask and the optional None reference-score value.
    from easydel.trainers.grpo_replay_buffer_trainer import GRPOWithReplayBufferTrainer
    from easydel.trainers.grpo_replay_buffer_trainer.replay_buffer import _ReplayBuffer

    replay = object.__new__(GRPOWithReplayBufferTrainer)
    replay.arguments = SimpleNamespace(num_generations=2)
    replay._pad_token_id = 0
    replay.replay_buffer = _ReplayBuffer(4, seed=0)
    _, inserted = replay._apply_replay_buffer(model_batch)
    assert inserted["replay_groups_added"] == 2
    flat_batch = {**model_batch, "advantages": jnp.zeros_like(model_batch["advantages"])}
    replaced_batch, replaced = replay._apply_replay_buffer(flat_batch)
    assert replaced["replay_groups_replaced"] == 2
    np.testing.assert_array_equal(np.asarray(replaced_batch["completion_attention_mask"]), np.ones((4, 2)))
    # With ref_state=None GRPO emits an all-zero reference placeholder (KL disabled);
    # replay must carry it through with the padded completion shape.
    np.testing.assert_array_equal(np.asarray(replaced_batch["ref_per_token_logps"]), np.zeros((4, 2)))


def test_tool_observations_remain_visible_to_attention_not_policy_loss():
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = SimpleNamespace(max_tool_calling_iterations=1)
    trainer._pad_token_id = 0
    trainer._tool_call_loop_enabled = lambda: True
    trainer._execute_tool_calls_as_observation = lambda calls: "observation"
    trainer._encode_tool_loop_fragment = lambda text: [8, 9]
    prompt = jnp.array([[1]])
    ids = jnp.array([[7, 0, 0, 0]])
    mask = jnp.array([[1, 0, 0, 0]])
    initial = GenerationResults(
        generation_results=["call"],
        prompt_ids=prompt,
        prompt_mask=jnp.ones_like(prompt),
        sequences=jnp.concatenate([prompt, ids], axis=1),
        completion_ids=ids,
        completion_mask=mask,
        decoded_prompts=["question"],
        completion_prompts=["question"],
        text=["call"],
        raw_text=["call"],
        tool_calls=[[{"function": {"name": "lookup"}}]],
        sampling_per_token_logps=jnp.array([[-0.25, 0.0, 0.0, 0.0]]),
    )
    followup = initial._replace(
        completion_ids=jnp.array([[10]]),
        completion_mask=jnp.array([[1]]),
        text=["answer"],
        raw_text=["answer"],
        tool_calls=[None],
        sampling_per_token_logps=jnp.array([[-0.75]]),
    )
    trainer.rollout = lambda **kwargs: followup
    result, loss_mask = trainer._maybe_run_tool_call_generation_loop(state=None, results=initial, completion_mask=mask)
    np.testing.assert_array_equal(np.asarray(result.completion_ids), [[7, 8, 9, 10]])
    np.testing.assert_array_equal(np.asarray(loss_mask), [[1, 0, 0, 1]])
    np.testing.assert_array_equal(np.asarray(result.completion_mask), [[1, 1, 1, 1]])
    np.testing.assert_allclose(np.asarray(result.sampling_per_token_logps), [[-0.25, 0.0, 0.0, -0.75]])
