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

"""End-to-end tests for ``TrainingArguments.eval_config_overrides``.

The production shape being covered: train a distillation run on one attention
mechanism (e.g. SPLASH at 131k) while the periodic in-training evaluation
(``do_eval`` + ``evaluation_steps``) runs the forward — student AND teacher —
on a different, reference mechanism (vanilla) at a shorter fixed sequence
length, without touching the training state.

Assertions target observable behavior, not wiring internals:

- the training loop fires evaluation exactly at the ``evaluation_steps``
  cadence;
- the state fed to the compiled eval step reconstructs (``state.model``) to a
  model whose config carries the override, for both student and teacher, while
  the trainer's own states keep the training config;
- a full eval pass leaves every parameter and optimizer buffer bit-identical;
- bad overrides fail loudly at trainer construction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

import easydel as ed
from easydel.utils.traversals import deepcopy_model

VOCAB = 64
SEQ_LEN = 16
BATCH = 8
TRAIN_ROWS = 32
EVAL_ROWS = 16
TRAIN_ATTN = "sdpa"
EVAL_ATTN = "vanilla"


class _Tokenizer:
    """Minimal tokenizer stand-in for pretokenized datasets (no network)."""

    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __len__(self):
        return VOCAB


def _tiny_student_state():
    """Tiny Llama built the production way: lazy_init + weight fill.

    ``eval_config_overrides`` (like training buckets) builds its variant
    graphdef via ``lazy_init``, so the base state must carry lazy-style graph
    bookkeeping — exactly what ``from_pretrained``/``load_state`` produce.
    ``materialize_meta_state`` zero-fills the lazy parameters; they are then
    replaced with deterministic random values.
    """
    config = ed.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attn_mechanism=TRAIN_ATTN,
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
    module = module.materialize_meta_state(seed=0)
    state = module.shard_model().to_state()
    rng = np.random.default_rng(0)
    graphstate = jax.tree_util.tree_map(
        lambda x: jnp.asarray(rng.normal(0.0, 0.02, x.shape), x.dtype),
        state.graphstate,
    )
    return state.replace(graphstate=graphstate)


def _token_rows(num_rows: int, *, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    ids = rng.integers(2, VOCAB, size=(num_rows, SEQ_LEN), dtype=np.int32)
    return Dataset.from_dict(
        {
            "input_ids": ids.tolist(),
            "attention_mask": np.ones((num_rows, SEQ_LEN), dtype=np.int32).tolist(),
        }
    )


def _make_trainer(tmp_path, *, eval_config_overrides, max_training_steps: int = 4):
    student_state = _tiny_student_state()
    teacher_state = deepcopy_model(student_state)
    arguments = ed.DistillationConfig(
        model_name="eval-config-override-test",
        save_directory=str(tmp_path),
        max_training_steps=max_training_steps,
        num_train_epochs=1,
        total_batch_size=BATCH,
        eval_batch_size=BATCH,
        do_eval=True,
        evaluation_steps=2,
        eval_config_overrides=eval_config_overrides,
        max_length=SEQ_LEN,
        learning_rate=1e-4,
        alpha=1.0,
        temperature=1.0,
        disable_dropout=True,
        use_wandb=False,
        do_last_save=False,
        save_steps=None,
        save_optimizer_state=False,
        log_steps=1_000_000,
        report_steps=1_000_000,
        weight_distribution_log_steps=0,
        track_memory=False,
        shuffle_train_dataset=False,
        progress_bar_type="json",
    )
    return ed.DistillationTrainer(
        arguments=arguments,
        processing_class=_Tokenizer(),
        student_model=student_state,
        teacher_model=teacher_state,
        train_dataset=_token_rows(TRAIN_ROWS, seed=0),
        eval_dataset=_token_rows(EVAL_ROWS, seed=1),
    )


def _attn_of(state) -> str:
    """Attention mechanism of the model a state reconstructs to."""
    return str(state.model.config.attn_mechanism)


class TestEvalConfigOverrideEndToEnd:
    def test_eval_cadence_override_and_state_isolation(self, tmp_path):
        trainer = _make_trainer(tmp_path, eval_config_overrides={"attn_mechanism": EVAL_ATTN})

        # -- Spy on the compiled eval step: capture the exact states it is fed.
        captured_student_states = []
        captured_teacher_states = []
        inner_eval_fn = trainer.sharded_evaluation_step_function

        def eval_fn_spy(state, batch, teacher_state, *static_args):
            captured_student_states.append(state)
            captured_teacher_states.append(teacher_state)
            return inner_eval_fn(state, batch, teacher_state, *static_args)

        trainer.sharded_evaluation_step_function = eval_fn_spy

        # -- Spy on eval invocations to observe the firing cadence.
        eval_fired_at_steps = []
        inner_eval = trainer.eval

        def eval_spy(model_state):
            eval_fired_at_steps.append(int(jax.device_get(model_state.step)))
            yield from inner_eval(model_state)

        trainer.eval = eval_spy

        output = trainer.train()

        # Cadence: evaluation_steps=2 over 4 training steps -> in-loop evals at
        # global steps 2 and 4; _finalize_training adds one final eval at 4.
        assert eval_fired_at_steps[:2] == [2, 4], eval_fired_at_steps
        assert all(step % 2 == 0 for step in eval_fired_at_steps)

        # Each eval pass consumes the full fixed eval set: EVAL_ROWS / BATCH
        # eval steps per pass, all through the compiled eval fn.
        eval_passes = len(eval_fired_at_steps)
        assert len(captured_student_states) == eval_passes * (EVAL_ROWS // BATCH)

        # Override reaches the compiled forward: the state fed to the eval fn
        # reconstructs to the overridden mechanism (student AND teacher) while
        # the trainer's own states keep the training mechanism.
        assert all(_attn_of(s) == EVAL_ATTN for s in captured_student_states)
        assert all(_attn_of(t) == EVAL_ATTN for t in captured_teacher_states)
        assert _attn_of(trainer.model_state) == TRAIN_ATTN
        assert _attn_of(trainer.teacher_state) == TRAIN_ATTN

        # Training completed all steps.
        final_step = int(jax.device_get(output.state.step))
        assert final_step == 4

    def test_eval_pass_leaves_training_state_bit_identical(self, tmp_path):
        trainer = _make_trainer(tmp_path, eval_config_overrides={"attn_mechanism": EVAL_ATTN})

        # Spy: the state fed to the compiled eval fn must SHARE the training
        # buffers (graphdef swap, not a parameter copy).
        captured_states = []
        inner_eval_fn = trainer.sharded_evaluation_step_function

        def eval_fn_spy(state, batch, teacher_state, *static_args):
            captured_states.append(state)
            return inner_eval_fn(state, batch, teacher_state, *static_args)

        trainer.sharded_evaluation_step_function = eval_fn_spy

        params_before = jax.device_get(jax.tree_util.tree_leaves(trainer.model_state.graphstate))
        opt_before = jax.device_get(jax.tree_util.tree_leaves(trainer.model_state.opt_state))
        step_before = int(jax.device_get(trainer.model_state.step))

        eval_metrics = list(trainer.eval(trainer.model_state))
        first_train_leaf = jax.tree_util.tree_leaves(trainer.model_state.graphstate)[0]
        for eval_state in captured_states:
            assert jax.tree_util.tree_leaves(eval_state.graphstate)[0] is first_train_leaf
        assert len(eval_metrics) == EVAL_ROWS // BATCH
        loss_keys = [k for k in eval_metrics[-1] if k.endswith("loss") and eval_metrics[-1][k] is not None]
        assert loss_keys, sorted(eval_metrics[-1])
        assert all(np.isfinite(float(eval_metrics[-1][k])) for k in loss_keys)

        params_after = jax.device_get(jax.tree_util.tree_leaves(trainer.model_state.graphstate))
        opt_after = jax.device_get(jax.tree_util.tree_leaves(trainer.model_state.opt_state))
        for before, after in zip(params_before, params_after, strict=True):
            np.testing.assert_array_equal(before, after)
        for before, after in zip(opt_before, opt_after, strict=True):
            np.testing.assert_array_equal(before, after)
        assert int(jax.device_get(trainer.model_state.step)) == step_before

    def test_override_and_no_override_evals_diverge_only_in_graphdef(self, tmp_path):
        """Same weights + same eval set: override changes the graphdef, not results.

        vanilla and sdpa compute the same math in f32 on CPU, so the eval loss
        must agree to numerical tolerance — catching overrides that silently
        rebuild parameters instead of swapping the graphdef.
        """
        trainer_plain = _make_trainer(tmp_path / "plain", eval_config_overrides=None)
        loss_plain = float(next(iter(trainer_plain.eval(trainer_plain.model_state)))["eval/loss"])

        trainer_override = _make_trainer(tmp_path / "override", eval_config_overrides={"attn_mechanism": EVAL_ATTN})
        loss_override = float(next(iter(trainer_override.eval(trainer_override.model_state)))["eval/loss"])

        assert trainer_plain._eval_graphdef is None
        assert trainer_override._eval_graphdef is not None
        np.testing.assert_allclose(loss_override, loss_plain, rtol=2e-3, atol=2e-3)


class TestEvalConfigOverrideGuards:
    def test_unknown_config_key_fails_at_construction(self, tmp_path):
        with pytest.raises(AttributeError, match="does_not_exist"):
            _make_trainer(tmp_path, eval_config_overrides={"does_not_exist": 1})

    def test_structure_changing_override_fails_at_construction(self, tmp_path):
        with pytest.raises(ValueError, match="parameter structure"):
            _make_trainer(tmp_path, eval_config_overrides={"num_hidden_layers": 1})
