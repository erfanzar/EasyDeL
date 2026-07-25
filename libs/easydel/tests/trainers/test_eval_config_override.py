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


def _token_rows(num_rows: int, *, seed: int, seq_len: int = SEQ_LEN) -> Dataset:
    rng = np.random.default_rng(seed)
    ids = rng.integers(2, VOCAB, size=(num_rows, seq_len), dtype=np.int32)
    return Dataset.from_dict(
        {
            "input_ids": ids.tolist(),
            "attention_mask": np.ones((num_rows, seq_len), dtype=np.int32).tolist(),
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


class TestDataloaderFastForwardOnResume:
    """`dataloader_fast_forward_on_resume=False` resumes without seeking the
    train stream — the escape hatch for pack-on-the-fly sources where a
    sequential skip re-renders the whole consumed prefix."""

    @pytest.mark.parametrize("fast_forward", [True, False])
    def test_resume_honors_fast_forward_flag(self, tmp_path, fast_forward):
        trainer = _make_trainer(tmp_path / str(fast_forward), eval_config_overrides=None)
        trainer.arguments.dataloader_fast_forward_on_resume = fast_forward
        resume_step = 2
        trainer.model_state = trainer.model_state.replace(
            step=jnp.asarray(resume_step, dtype=trainer.model_state.step.dtype)
        )

        forwarded = []
        inner_fast_forward = trainer._fast_forward_batches

        def fast_forward_spy(data_iter, dataloader, num_batches):
            forwarded.append(int(num_batches))
            return inner_fast_forward(data_iter, dataloader, num_batches)

        trainer._fast_forward_batches = fast_forward_spy

        output = trainer.train()
        assert int(jax.device_get(output.state.step)) == 4
        if fast_forward:
            assert forwarded == [resume_step]
        else:
            assert forwarded == []


# 2:1 training-bucket cadence (mirrors the production v2 recipe: 2 x long-attn :
# 1 x short-vanilla) combined with the periodic 8k-vanilla eval override.
LONG_LEN = 16
SHORT_LEN = 8
BUCKET_BATCH = 8


def _make_bucketed_trainer_with_eval(tmp_path, *, max_training_steps: int = 6):
    """DistillationTrainer with two TRAINING buckets + periodic eval override.

    bucket 0 = "long" (base attn = sdpa, LONG_LEN), bucket 1 = "short"
    (per-bucket attn override = vanilla, SHORT_LEN). ModBucketRule mod=3 fires
    the short bucket once per three steps (2:1). Eval (evaluation_steps) runs a
    separate held-out set through the vanilla eval-config override — a distinct
    path from the vanilla TRAINING bucket.
    """
    from easydel.trainers.buckets import ModBucketRule, TrainingBucket

    student_state = _tiny_student_state()
    teacher_state = deepcopy_model(student_state)
    arguments = ed.DistillationConfig(
        model_name="bucket-eval-coexist-test",
        save_directory=str(tmp_path),
        max_training_steps=max_training_steps,
        num_train_epochs=1,
        total_batch_size=BUCKET_BATCH,
        eval_batch_size=BUCKET_BATCH,
        do_eval=True,
        evaluation_steps=3,
        eval_config_overrides={"attn_mechanism": EVAL_ATTN},
        max_length=LONG_LEN,
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
    buckets = [
        TrainingBucket(name="long", max_length=LONG_LEN, config=None, total_batch_size=BUCKET_BATCH),
        TrainingBucket(
            name="short",
            max_length=SHORT_LEN,
            config={"attn_mechanism": EVAL_ATTN},
            total_batch_size=BUCKET_BATCH,
        ),
    ]
    # (step-2)%3==0 -> short(1); else long(0): steps 0,1->long 2->short 3,4->long 5->short.
    bucket_rule = ModBucketRule(mod=3, offset=2, on_bucket=1, off_bucket=0)
    # Enough rows that the resolved step capacity covers max_training_steps
    # (rows / BUCKET_BATCH must be >= max_training_steps).
    bucket_train_rows = BUCKET_BATCH * max_training_steps * 2
    bucket_datasets = [
        _token_rows(bucket_train_rows, seed=0, seq_len=LONG_LEN),
        _token_rows(bucket_train_rows, seed=2, seq_len=SHORT_LEN),
    ]
    return ed.DistillationTrainer(
        arguments=arguments,
        processing_class=_Tokenizer(),
        student_model=student_state,
        teacher_model=teacher_state,
        train_dataset=_token_rows(bucket_train_rows, seed=0, seq_len=LONG_LEN),
        eval_dataset=_token_rows(EVAL_ROWS, seed=1, seq_len=LONG_LEN),
        buckets=buckets,
        bucket_rule=bucket_rule,
        bucket_datasets=bucket_datasets,
    )


class TestTrainingBucketsCoexistWithEvalOverride:
    """The 2:1 training buckets (per-bucket attn) and the eval-config override
    are independent: training swaps per-bucket graphdefs, eval swaps its own."""

    def test_per_bucket_attn_and_eval_override_are_independent(self, tmp_path):
        trainer = _make_bucketed_trainer_with_eval(tmp_path)

        # Two training buckets, each with its own graphdef (built once).
        assert len(trainer._bucket_graphdefs) == 2
        long_attn = _attn_of(trainer.model_state.replace(graphdef=trainer._bucket_graphdefs[0]))
        short_attn = _attn_of(trainer.model_state.replace(graphdef=trainer._bucket_graphdefs[1]))
        assert long_attn == TRAIN_ATTN  # base attn (sdpa)
        assert short_attn == EVAL_ATTN  # per-bucket vanilla override
        # The eval override graphdef exists and is a DISTINCT object from the
        # vanilla training-bucket graphdef (separate compile, separate purpose).
        assert trainer._eval_graphdef is not None
        assert trainer._eval_graphdef is not trainer._bucket_graphdefs[1]
        assert _attn_of(trainer.model_state.replace(graphdef=trainer._eval_graphdef)) == EVAL_ATTN

    def test_bucket_step_fns_compiled_once_and_reused(self, tmp_path):
        trainer = _make_bucketed_trainer_with_eval(tmp_path)

        # Per-bucket compiled step fns are built once at init (before train()),
        # one per bucket — this is the cache the runtime loop indexes into.
        assert set(trainer._bucket_step_fns) == {0, 1}

        # Wrap the cached fns with per-bucket counters. The training loop only
        # INDEXES self._bucket_step_fns[bi]; it never recompiles. So a single
        # cached fn per bucket must service every step routed to that bucket.
        calls = {0: 0, 1: 0}

        def _wrap(idx, fn):
            def wrapped(*args, **kwargs):
                calls[idx] += 1
                return fn(*args, **kwargs)

            return wrapped

        trainer._bucket_step_fns[0] = _wrap(0, trainer._bucket_step_fns[0])
        trainer._bucket_step_fns[1] = _wrap(1, trainer._bucket_step_fns[1])

        output = trainer.train()

        # 6 steps at 2:1 -> long bucket 4 steps, short bucket 2 steps, each
        # served by its one cached compiled fn (no per-switch recompilation).
        assert int(jax.device_get(output.state.step)) == 6
        assert calls == {0: 4, 1: 2}, calls

    def test_eval_runs_vanilla_and_training_state_survives_under_buckets(self, tmp_path):
        trainer = _make_bucketed_trainer_with_eval(tmp_path)

        captured_eval_states = []
        inner_eval_fn = trainer.sharded_evaluation_step_function

        def eval_fn_spy(state, batch, teacher_state, *static_args):
            captured_eval_states.append((state, teacher_state))
            return inner_eval_fn(state, batch, teacher_state, *static_args)

        trainer.sharded_evaluation_step_function = eval_fn_spy

        params_before = jax.device_get(jax.tree_util.tree_leaves(trainer.model_state.graphstate))
        output = trainer.train()

        # Eval fired during the bucketed run and used the vanilla override for
        # both student and teacher forwards.
        assert captured_eval_states, "eval never fired under training buckets"
        for student_state, teacher_state in captured_eval_states:
            assert _attn_of(student_state) == EVAL_ATTN
            assert _attn_of(teacher_state) == EVAL_ATTN

        # Training completed all 6 steps; the base (training) config is intact.
        assert int(jax.device_get(output.state.step)) == 6
        assert _attn_of(trainer.model_state) == TRAIN_ATTN
        assert _attn_of(trainer.teacher_state) == TRAIN_ATTN
        # Params moved (training happened) but stayed finite.
        params_after = jax.device_get(jax.tree_util.tree_leaves(output.state.graphstate))
        assert all(np.all(np.isfinite(p)) for p in params_after)
        assert len(params_after) == len(params_before)


# Three-bucket 2:2:1 cadence (production v3 recipe: 2 x medium-vanilla,
# 2 x short-vanilla, 1 x long-base-attn over mod 5).
MEDIUM_LEN = 12


def _make_three_bucket_trainer(tmp_path, *, max_training_steps: int = 10, resume_step: int = 0):
    """DistillationTrainer with THREE training buckets on a 2:2:1 PatternBucketRule.

    bucket 0 = medium (vanilla override, MEDIUM_LEN), bucket 1 = short (vanilla
    override, SHORT_LEN), bucket 2 = long (base attn, LONG_LEN). The pattern
    [0,0,1,1,2] fires 2 medium : 2 short : 1 long per 5-step cycle. The periodic
    eval override runs on its own fixed set, independent of all three.
    """
    from easydel.trainers.buckets import PatternBucketRule, TrainingBucket

    student_state = _tiny_student_state()
    teacher_state = deepcopy_model(student_state)
    arguments = ed.DistillationConfig(
        model_name="three-bucket-test",
        save_directory=str(tmp_path),
        max_training_steps=max_training_steps,
        num_train_epochs=1,
        total_batch_size=BUCKET_BATCH,
        eval_batch_size=BUCKET_BATCH,
        do_eval=True,
        evaluation_steps=5,
        eval_config_overrides={"attn_mechanism": EVAL_ATTN},
        max_length=LONG_LEN,
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
        step_start_point=resume_step or None,
        force_step_start_point=bool(resume_step),
    )
    buckets = [
        TrainingBucket(
            name="medium",
            max_length=MEDIUM_LEN,
            config={"attn_mechanism": EVAL_ATTN},
            total_batch_size=BUCKET_BATCH,
        ),
        TrainingBucket(
            name="short",
            max_length=SHORT_LEN,
            config={"attn_mechanism": EVAL_ATTN},
            total_batch_size=BUCKET_BATCH,
        ),
        TrainingBucket(name="long", max_length=LONG_LEN, config=None, total_batch_size=BUCKET_BATCH),
    ]
    bucket_rule = PatternBucketRule(pattern=[0, 0, 1, 1, 2])
    bucket_train_rows = BUCKET_BATCH * (max_training_steps + resume_step) * 2
    bucket_datasets = [
        _token_rows(bucket_train_rows, seed=3, seq_len=MEDIUM_LEN),
        _token_rows(bucket_train_rows, seed=2, seq_len=SHORT_LEN),
        _token_rows(bucket_train_rows, seed=0, seq_len=LONG_LEN),
    ]
    return ed.DistillationTrainer(
        arguments=arguments,
        processing_class=_Tokenizer(),
        student_model=student_state,
        teacher_model=teacher_state,
        train_dataset=_token_rows(bucket_train_rows, seed=0, seq_len=LONG_LEN),
        eval_dataset=_token_rows(EVAL_ROWS, seed=1, seq_len=LONG_LEN),
        buckets=buckets,
        bucket_rule=bucket_rule,
        bucket_datasets=bucket_datasets,
    )


class TestThreeBucketPatternCadence:
    """2:2:1 over three buckets with per-bucket attn + shapes, plus eval."""

    def test_three_graphdefs_with_correct_per_bucket_attn_and_length(self, tmp_path):
        trainer = _make_three_bucket_trainer(tmp_path)

        assert len(trainer._bucket_graphdefs) == 3
        attns = [_attn_of(trainer.model_state.replace(graphdef=trainer._bucket_graphdefs[i])) for i in range(3)]
        assert attns == [EVAL_ATTN, EVAL_ATTN, TRAIN_ATTN]
        # Each bucket keeps its own sequence length (fixed shape per program).
        assert [b.max_length for b in trainer._buckets] == [MEDIUM_LEN, SHORT_LEN, LONG_LEN]
        # Three distinct training programs + one distinct eval program.
        assert len({id(g) for g in trainer._bucket_graphdefs.values()}) == 3
        assert trainer._eval_graphdef is not None
        assert all(trainer._eval_graphdef is not g for g in trainer._bucket_graphdefs.values())

    def test_realized_2_2_1_step_ratio_and_one_compile_per_bucket(self, tmp_path):
        trainer = _make_three_bucket_trainer(tmp_path, max_training_steps=10)

        # One cached compiled step fn per bucket, built before train().
        assert set(trainer._bucket_step_fns) == {0, 1, 2}

        calls = {0: 0, 1: 0, 2: 0}
        batch_shapes = {0: set(), 1: set(), 2: set()}

        def _wrap(idx, fn):
            def wrapped(state, batch, *args, **kwargs):
                calls[idx] += 1
                batch_shapes[idx].add(tuple(np.asarray(batch["input_ids"]).shape))
                return fn(state, batch, *args, **kwargs)

            return wrapped

        for i in range(3):
            trainer._bucket_step_fns[i] = _wrap(i, trainer._bucket_step_fns[i])

        output = trainer.train()

        # 10 steps = 2 full cycles -> 4 medium : 4 short : 2 long (exactly 2:2:1).
        assert int(jax.device_get(output.state.step)) == 10
        assert calls == {0: 4, 1: 4, 2: 2}, calls
        # Each bucket's program saw exactly ONE batch shape (fixed shapes => the
        # single cached compile per bucket is never re-traced).
        assert batch_shapes[0] == {(BUCKET_BATCH, MEDIUM_LEN)}, batch_shapes[0]
        assert batch_shapes[1] == {(BUCKET_BATCH, SHORT_LEN)}, batch_shapes[1]
        assert batch_shapes[2] == {(BUCKET_BATCH, LONG_LEN)}, batch_shapes[2]

    def test_cadence_phase_is_anchored_on_the_resumed_global_step(self, tmp_path):
        # Resuming at a step divisible by the modulus must open the cycle at
        # phase 0 (two medium steps), matching the production 3000-resume.
        resume_step = 10
        trainer = _make_three_bucket_trainer(tmp_path, max_training_steps=15, resume_step=resume_step)
        assert int(jax.device_get(trainer.model_state.step)) == resume_step

        order = []
        for i in range(3):
            inner = trainer._bucket_step_fns[i]

            def _wrap(idx, fn):
                def wrapped(*args, **kwargs):
                    order.append(idx)
                    return fn(*args, **kwargs)

                return wrapped

            trainer._bucket_step_fns[i] = _wrap(i, inner)

        trainer.train()
        # Steps 10..14 are one full cycle starting at phase 0.
        assert order[:5] == [0, 0, 1, 1, 2], order[:5]
