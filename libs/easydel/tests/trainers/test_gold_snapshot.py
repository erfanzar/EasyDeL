"""Actual GOLD initialization, eval variants, and buffer ownership (accelerator-only)."""

import easydel as ed
import jax
import numpy as np
import pytest
import spectrax as spx
from datasets import Dataset
from jax import numpy as jnp


class Tokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __len__(self):
        return 32


def make_student():
    config = ed.LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attn_mechanism="sdpa",
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
    )
    config.attn_dtype = jnp.float32
    module = ed.LlamaForCausalLM.lazy_init(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    return module.materialize_meta_state(seed=0).shard_model()


@pytest.mark.parametrize("use_state", [False, True])
@pytest.mark.parametrize("eval_overrides", [None, {"attn_mechanism": "vanilla"}])
def test_gold_self_teacher_initialization_and_donation(tmp_path, use_state, eval_overrides):
    student = make_student()
    dataset = Dataset.from_dict({"input_ids": [[2, 3, 4, 5]] * 4, "attention_mask": [[1, 1, 1, 1]] * 4})
    arguments = ed.GOLDConfig(
        model_name="gold-owned-teacher",
        save_directory=str(tmp_path),
        max_training_steps=1,
        total_batch_size=4,
        eval_batch_size=4,
        max_length=4,
        alpha=1.0,
        temperature=1.0,
        disable_dropout=False,
        eval_config_overrides=eval_overrides,
        use_wandb=False,
        do_last_save=False,
        save_steps=None,
        save_optimizer_state=False,
        weight_distribution_log_steps=0,
        track_memory=False,
        shuffle_train_dataset=False,
        progress_bar_type="json",
    )
    # Neither GOLD nor DistillationTrainer initialization is mocked: exercise
    # the actual mode swap, eval override construction, and compilation setup.
    trainer = ed.GOLDTrainer(
        arguments=arguments,
        model=student.to_state() if use_state else student,
        processing_class=Tokenizer(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    teacher = trainer.teacher_state
    eval_teacher = trainer._eval_shared_fn_extra_args[0]
    assert trainer.model_state.model._spx_training
    assert not teacher.model._spx_training
    assert not eval_teacher.model._spx_training
    assert teacher.tx is None
    assert teacher.opt_state is None
    assert str(teacher.model.config.attn_mechanism) == "sdpa"
    expected_attention = "vanilla" if eval_overrides else "sdpa"
    assert str(eval_teacher.model.config.attn_mechanism) == expected_attention

    expected_weights = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(teacher.graphstate)]
    student_leaves = jax.tree_util.tree_leaves((trainer.model_state.graphstate, trainer.model_state.graphother))
    teacher_leaves = jax.tree_util.tree_leaves((teacher.graphstate, teacher.graphother))
    assert len(student_leaves) == len(teacher_leaves)
    assert all(a is not b for a, b in zip(student_leaves, teacher_leaves, strict=True))
    # Eval overrides must share the OWNED teacher buffers, never student buffers.
    assert all(
        a is b
        for a, b in zip(
            teacher_leaves,
            jax.tree_util.tree_leaves((eval_teacher.graphstate, eval_teacher.graphother)),
            strict=True,
        )
    )

    updated_weights = jax.jit(lambda tree: jax.tree_util.tree_map(lambda weight: weight + 1, tree), donate_argnums=(0,))(
        trainer.model_state.graphstate
    )
    trainer.model_state = trainer.model_state.replace(graphstate=updated_weights)
    for actual, expected in zip(jax.tree_util.tree_leaves(teacher.graphstate), expected_weights, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(jax.tree_util.tree_leaves(eval_teacher.graphstate), expected_weights, strict=True):
        np.testing.assert_array_equal(actual, expected)

    # Execute the configured evaluator: its input shardings include static
    # teacher graph metadata and must agree with the state chosen at call time.
    eval_state = trainer.model_state
    if eval_overrides:
        eval_state = eval_state.replace(graphdef=trainer._eval_graphdef)
    batch = {"input_ids": jnp.asarray([[2, 3, 4, 5]] * 4), "attention_mask": jnp.ones((4, 4), dtype=jnp.int32)}
    metrics = trainer.sharded_evaluation_step_function(
        eval_state, batch, *trainer._eval_shared_fn_extra_args, *trainer._eval_shared_fn_static_args
    )
    assert np.isfinite(np.asarray(metrics.loss))
