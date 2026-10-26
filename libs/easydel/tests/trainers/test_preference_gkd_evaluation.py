"""Configured evaluation integration for policy/teacher static variants (accelerator only)."""

import easydel as ed
import jax
import numpy as np
import pytest
from datasets import Dataset
from jax import numpy as jnp

from .test_gold_snapshot import Tokenizer, make_student


@pytest.mark.parametrize("family", ["cpo", "dpo", "gkd"])
def test_configured_preference_and_gkd_eval_overrides(tmp_path, family):
    student = make_student()
    arguments = dict(
        model_name=f"r5-{family}-eval",
        save_directory=str(tmp_path),
        max_training_steps=1,
        total_batch_size=4,
        eval_batch_size=4,
        max_length=4,
        eval_config_overrides={"attn_mechanism": "vanilla"},
        use_wandb=False,
        do_last_save=False,
        save_steps=None,
        save_optimizer_state=False,
        weight_distribution_log_steps=0,
        track_memory=False,
        shuffle_train_dataset=False,
        progress_bar_type="json",
    )
    if family == "gkd":
        config = ed.GKDConfig(**arguments, lmbda=0.0, max_new_tokens=1)
        cls = ed.GKDTrainer
        examples = {
            "input_ids": [[2, 3, 4, 5]] * 4,
            "attention_mask": [[1, 1, 1, 1]] * 4,
            "labels": [[-100, -100, 4, 5]] * 4,
        }
    else:
        arguments.update(max_prompt_length=2, max_completion_length=2)
        config = ed.CPOConfig(**arguments, cpo_alpha=0.0) if family == "cpo" else ed.DPOConfig(**arguments)
        cls = ed.CPOTrainer if family == "cpo" else ed.DPOTrainer
        # Equal chosen/rejected streams imply a zero log-ratio differential,
        # independent of model weights. DPO/CPO sigmoid losses are log(2).
        examples = {"prompt_input_ids": [[2, 3]] * 4, "prompt_attention_mask": [[1, 1]] * 4}
        for prefix in ("chosen", "rejected"):
            examples[f"{prefix}_input_ids"] = [[4, 5]] * 4
            examples[f"{prefix}_attention_mask"] = [[1, 1]] * 4
    dataset = Dataset.from_dict(examples)
    trainer = cls(
        model=student,
        arguments=config,
        processing_class=Tokenizer(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    training_graphdef = trainer.model_state.graphdef
    state = trainer._eval_state_for_compiled_step(trainer.model_state)
    batch = {key: jnp.asarray(value) for key, value in examples.items()}
    metrics = jax.block_until_ready(
        trainer.sharded_evaluation_step_function(
            state,
            batch,
            *trainer._eval_shared_fn_extra_args,
            *trainer._eval_shared_fn_static_args,
        )
    )
    expected = 0.0 if family == "gkd" else np.log(2)
    np.testing.assert_allclose(metrics.loss, expected, rtol=2e-5, atol=2e-5)
    assert str(state.model.config.attn_mechanism) == "vanilla"
    if family != "cpo":
        assert str(trainer._eval_shared_fn_extra_args[0].model.config.attn_mechanism) == "vanilla"
    assert trainer.model_state.graphdef is training_graphdef
    assert str(trainer.model_state.model.config.attn_mechanism) == "sdpa"
