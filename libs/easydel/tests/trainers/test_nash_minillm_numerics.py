"""Accelerator-only numerical checks of Nash/MiniLLM objectives against independent references."""

import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.trainers.base_trainer import GenerationResults
from easydel.trainers.group_filtered_policy_optimization import GFPOTrainer
from easydel.trainers.group_relative_policy_optimization import GRPOTrainer
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step
from easydel.trainers.metrics import MetricsHistogram
from easydel.trainers.minillm_trainer import MiniLLMTrainer
from easydel.trainers.nash_md_trainer._fn import nash_md_step
from easydel.trainers.training_utils import ScheduledStepCall, get_scheduled_loss_adapter
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("policy_probs", [[0.8, 0.2], [0.6, 0.4]])
def test_nash_sampled_kl_gradient_matches_exact_kl(scheduled, policy_probs):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    probs = np.asarray(policy_probs, dtype=np.float32)
    reference_probs = np.asarray([0.8, 0.2], dtype=np.float32)
    tokens = jnp.asarray([[0]] * round(5 * probs[0]) + [[1]] * round(5 * probs[1]))
    batch = {
        "prompt_ids": jnp.zeros((5, 1), dtype=jnp.int32),
        "prompt_mask": jnp.ones((5, 1), dtype=jnp.int32),
        "completion_ids": tokens,
        "completion_mask": jnp.ones((5, 1), dtype=jnp.int32),
        "ref_token_logps": jnp.log(jnp.asarray(reference_probs))[tokens],
        "probabilities": jnp.full((5,), 0.5),
    }

    class Model:
        def __init__(self, logits):
            self.mesh = mesh
            self.logits = logits

        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=jnp.broadcast_to(self.logits, (*input_ids.shape, 2)))

    class State:
        graphdef = None
        graphother = None

        def __init__(self, logits):
            self.graphstate = logits
            self.model = Model(logits)

        def merge(self, tree):
            return Model(tree)

    def objective(logits):
        state = State(logits)
        args = (state, batch, 0.3, None, None, None, PartitionSpec(), 1, False)
        if not scheduled:
            return nash_md_step(*args).loss
        bound = inspect.signature(nash_md_step).bind(*args)
        bound.apply_defaults()
        call = ScheduledStepCall(nash_md_step, state, batch, args, {}, bound.arguments, None)
        return get_scheduled_loss_adapter(nash_md_step).make_loss(call)(logits, batch)

    actual = jax.grad(objective)(jnp.log(jnp.asarray(probs)))
    log_ratio = np.log(probs) - np.log(reference_probs)
    expected = 0.3 * probs * (log_ratio - np.sum(probs * log_ratio))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("length", [2, 128])
def test_minillm_discounted_suffix_is_relative_to_current_token(normalize, length):
    rewards = np.ones((2, length), dtype=np.float32)
    rewards[1] = np.linspace(-1.0, 1.0, length)
    mask = np.ones_like(rewards)
    mask[1, 1::3] = 0
    actual = MiniLLMTrainer._compute_reverse_kl_advantage(
        student_logps=jnp.zeros_like(jnp.asarray(rewards)),
        teacher_logps=jnp.asarray(rewards),
        completion_mask=jnp.asarray(mask),
        gamma=0.5,
        length_normalization=normalize,
    )
    expected = np.zeros_like(rewards)
    for row in range(2):
        for t in range(length):
            weights = np.asarray([0.5 ** (k - t) for k in range(t, length)])
            expected[row, t] = np.sum(weights * rewards[row, t:] * mask[row, t:])
            if normalize:
                expected[row, t] /= max(np.sum(weights * mask[row, t:]), 1e-4)
    expected *= mask
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("values", [np.ones(70001), np.asarray([300.0, 301.0])])
def test_fp16_histogram_statistics_remain_finite(values):
    actual = MetricsHistogram.from_array(jnp.asarray(values, dtype=jnp.float16))
    reference = values.astype(np.float16).astype(np.float64)
    assert actual.sum.dtype == jnp.float32
    assert actual.sum_squares.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(actual.mean), reference.mean(), atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual.variance), reference.var(), atol=1e-6)
    assert np.isfinite(np.asarray(actual.std)).all()


def minimal_trainer(cls, reward):
    trainer = object.__new__(cls)
    trainer.arguments = SimpleNamespace(
        num_generations=1,
        num_generations_eval=1,
        mask_truncated_completions=False,
        max_prompt_length=1,
        max_length=5,
        esurge_importance_sampling_correction=False,
        logprob_vocab_chunk_size=None,
        advantage_estimator="group_relative",
        reward_clip_range=None,
        rkl_advantage=True,
        gamma=0.0,
        length_normalization=False,
        kd_temperature=1.0,
        single_step_decomposition=False,
    )
    trainer.steps_per_generation = 1
    trainer.num_iterations = 1
    trainer.ref_state = None
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.scale_rewards = "none"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.reward_weights = jnp.ones((1,))
    trainer.reward_funcs = [reward]
    trainer.reward_processing_classes = [None]
    trainer.reward_func_names = ["reward"]
    trainer._pad_token_id = 0
    trainer._eos_token_id = 2
    trainer._apply_user_data_collator = lambda batch: batch
    trainer._extract_reward_batch_sidechannels = lambda batch: {}
    trainer._purify_batch = lambda batch: batch
    trainer._all_gather = lambda value: value
    trainer._tool_call_loop_enabled = lambda: False
    trainer._run_environment_feedback = lambda **kwargs: None
    trainer._maybe_log_grpo_completions = lambda **kwargs: None
    trainer._log_training_generations_to_wandb = lambda **kwargs: None
    return trainer


def rollout(ids, mask):
    prompt = jnp.ones((1, 1), dtype=jnp.int32)
    return GenerationResults(
        generation_results=["answer"],
        prompt_ids=prompt,
        prompt_mask=jnp.ones_like(prompt),
        sequences=jnp.concatenate([prompt, ids], axis=1),
        completion_ids=ids,
        completion_mask=mask,
        decoded_prompts=["question"],
        completion_prompts=["question"],
        text=["answer"],
        raw_text=["answer"],
    )


@pytest.mark.parametrize("tool_turn", [False, True])
def test_grpo_reward_payload_and_loss_follow_actual_generated_tokens(tool_turn):
    seen = []

    def reward(completion_ids, completion_length, **kwargs):
        seen.append((completion_ids, completion_length))
        return [1.0]

    trainer = minimal_trainer(GRPOTrainer, reward)
    ids = jnp.asarray([[7, 0, 0, 0]])
    valid = jnp.asarray([[1, 0, 0, 0]])
    result = rollout(ids, valid)
    if tool_turn:
        trainer.arguments.max_tool_calling_iterations = 1
        trainer._tool_call_loop_enabled = lambda: True
        trainer._execute_tool_calls_as_observation = lambda calls: "observation"
        trainer._encode_tool_loop_fragment = lambda text: [8, 9]
        result = result._replace(tool_calls=[[{"function": {"name": "lookup"}}]])
        followup = rollout(jnp.asarray([[10]]), jnp.asarray([[1]]))
        outputs = iter([result, followup])
        trainer.rollout = lambda **kwargs: next(outputs)
    else:
        trainer.rollout = lambda **kwargs: result
    state = SimpleNamespace(model=lambda **kwargs: None, step=jnp.asarray(0))
    batch = {"input_ids": result.prompt_ids, "attention_mask": result.prompt_mask}
    model_batch, _ = trainer._preprocess_batch_input(state, batch, True)
    expected_mask = [[1, 0, 0, 1]] if tool_turn else [[1, 0, 0, 0]]
    expected_ids = [[7, 10]] if tool_turn else [[7]]
    np.testing.assert_array_equal(np.asarray(model_batch["completion_mask"]), expected_mask)
    assert int(model_batch["num_items_in_batch"]) == len(expected_ids[0])
    assert seen == [(expected_ids, [len(expected_ids[0])])]

    # Padding/observations contribute no policy gradient, even when their logits
    # differ strongly from those of sampled tokens.
    class Model:
        mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))

        def __init__(self, logits):
            self.logits = logits

        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.logits)

    class State:
        def __init__(self, logits):
            self.model = Model(logits)
            self.graphstate = logits

        def merge(self, tree):
            return Model(tree)

    model_batch["advantages"] = jnp.ones((1,))

    def objective(logits):
        return grpo_step(State(logits), model_batch, 1, 0.0, partition_spec=PartitionSpec(), is_training=False).loss

    logits = jnp.zeros((1, 5, 12), dtype=jnp.float32)
    gradients = jax.grad(objective)(logits)
    policy_mask = np.asarray(expected_mask)[0].astype(bool)
    np.testing.assert_array_equal(np.asarray(gradients)[0, :4][~policy_mask], 0.0)
    assert np.any(np.asarray(gradients)[0, :4][policy_mask] != 0)


@pytest.mark.parametrize("generation_factor", [1, 2])
def test_minillm_reverse_kl_rescoring_preserves_prompt_embeddings(generation_factor):
    from easydel.trainers.group_relative_policy_optimization._fn import get_per_token_logps

    trainer = minimal_trainer(MiniLLMTrainer, lambda **kwargs: [0.0] * generation_factor)
    trainer.arguments.num_generations = generation_factor
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))

    class Model:
        def __init__(self, scale):
            self.mesh = mesh
            self.scale = scale

        def compute_embedding(self, input_ids):
            return input_ids[..., None].astype(jnp.float32)

        def __call__(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
            # Prompt embeddings, not token-lookup embeddings, determine the
            # logits predicting the first completion token.
            logits = jnp.zeros((*attention_mask.shape, 4))
            signal = inputs_embeds[:, 0, 0] if inputs_embeds is not None else input_ids[:, 0].astype(jnp.float32)
            return SimpleNamespace(logits=logits.at[:, 0, 3].set(self.scale * signal))

    trainer.teacher_state = SimpleNamespace(model=Model(2.0))
    student = SimpleNamespace(model=Model(1.0), step=jnp.asarray(0))
    trainer.compute_state_logps = lambda state, ids, mask, kwargs, **options: get_per_token_logps(
        state.model, ids, mask, options["prompt_length"], model_kwargs=kwargs
    )
    result = rollout(jnp.asarray([[3]]), jnp.asarray([[1]]))
    result = result._replace(
        completion_ids=result.completion_ids.repeat(generation_factor, 0),
        completion_mask=result.completion_mask.repeat(generation_factor, 0),
        sequences=result.sequences.repeat(generation_factor, 0),
        generation_results=result.generation_results * generation_factor,
        completion_prompts=result.completion_prompts * generation_factor,
        text=result.text * generation_factor,
        raw_text=result.raw_text * generation_factor,
    )
    trainer.rollout = lambda **kwargs: result
    batch, _ = trainer._preprocess_batch_input(
        student,
        {"input_ids": result.prompt_ids, "attention_mask": result.prompt_mask, "inputs_embeds": jnp.asarray([[[3.0]]])},
        True,
    )
    # Independent two-way teacher/student categorical scores: logits [0,0,0,6]
    # versus [0,0,0,3], not token-id-conditioned logits [...,2] and [...,1].
    expected = (6 - np.logaddexp(np.log(3), 6)) - (3 - np.logaddexp(np.log(3), 3))
    np.testing.assert_allclose(np.asarray(batch["advantages"]), [[expected]] * generation_factor, atol=1e-6)


def test_gfpo_legacy_filter_and_shared_hook_select_identical_rows():
    trainer = object.__new__(GFPOTrainer)
    trainer.arguments = SimpleNamespace(num_remains_in_group=1)
    trainer.group_filter_func = lambda ids, rewards, mask: rewards
    ids = jnp.asarray([[7], [8], [9], [10]])
    mask = jnp.ones_like(ids)
    rewards = jnp.asarray([1.0, 2.0, 4.0, 3.0])
    selected = trainer._select_rollout_indices(ids, mask, rewards, 2)
    legacy = trainer._filter_completions(
        None, None, ids, mask, rewards[:, None], rewards, rewards[:, None], ["a", "b", "c", "d"], 2, 1
    )
    np.testing.assert_array_equal(np.asarray(ids[selected]), [[8], [9]])
    np.testing.assert_array_equal(np.asarray(legacy[2]), [[8], [9]])
    assert legacy[7] == ["b", "c"]
