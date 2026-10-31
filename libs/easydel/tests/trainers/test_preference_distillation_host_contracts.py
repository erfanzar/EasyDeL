"""Host-side trainer contracts; no JAX array computation is performed."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.trainers.contrastive_preference_optimization_trainer import cpo_trainer as cpo
from easydel.trainers.direct_preference_optimization_trainer import dpo_trainer as dpo
from easydel.trainers.generalized_knowledge_distillation_trainer import gkd_trainer as gkd
from easydel.trainers.trainer import Trainer

pytestmark = pytest.mark.cpu_ok


@dataclass(frozen=True)
class State:
    graphdef: str
    tx: str | None = None
    opt_state: object = None
    model: object = None

    @property
    def shardings(self):
        return self

    def replace(self, **changes):
        return replace(self, **changes)


class Arguments(SimpleNamespace):
    def __getattr__(self, name):
        # Unused optional loss/config fields in compiler-boundary tests.
        return None


def compile_contract(fn, *, in_shardings, **kwargs):
    def compiled(state, batch, *extra):
        if (state.graphdef, state.tx, state.opt_state) != (
            in_shardings[0].graphdef,
            in_shardings[0].tx,
            in_shardings[0].opt_state,
        ):
            raise ValueError("incompatible static state or optimizer metadata")
        for actual, expected in zip(extra, in_shardings[2:], strict=False):
            if isinstance(expected, State) and actual.graphdef != expected.graphdef:
                raise ValueError("incompatible auxiliary state metadata")
        return fn(state, batch, *extra)

    return compiled


@pytest.mark.parametrize(
    "module,cls,aux_name",
    [(cpo, cpo.CPOTrainer, None), (dpo, dpo.DPOTrainer, "reference"), (gkd, gkd.GKDTrainer, "teacher")],
)
def test_evaluation_compiles_for_runtime_policy_and_auxiliary_graphs(monkeypatch, module, cls, aux_name):
    monkeypatch.setattr(module, "replicated_named_sharding", lambda mesh: "replicated")
    monkeypatch.setattr(module, "resolve_straight_through_emulator", lambda **kwargs: None)
    monkeypatch.setattr(module, "compile_trainer_step", compile_contract)
    if module is dpo:
        monkeypatch.setattr(module, "compile_trainer_auxiliary", lambda fn, **kwargs: fn)
    step_name = "gkd_step" if module is gkd else "evaluation_step"
    monkeypatch.setattr(module, step_name, lambda state, batch, *aux: (state.graphdef, tuple(s.graphdef for s in aux)))
    trainer = object.__new__(cls)
    trainer._model = SimpleNamespace(mesh=None)
    trainer.arguments = Arguments(
        beta=0.5,
        temperature=1.0,
        ensure_checkpoint_path=lambda: None,
        get_streaming_checkpointer=lambda: None,
    )
    trainer.padding_value = 0
    trainer.scheduler = None
    trainer.state_shardings = State("training", tx="adam", opt_state="slots")
    trainer.model_state = trainer.state_shardings
    trainer._eval_graphdef = "eval-policy"
    trainer._runtime_trace = lambda *args: None
    if aux_name:
        auxiliary = State("train-aux", model=SimpleNamespace(flops_per_token=lambda **kwargs: 1))
        setattr(trainer, f"{aux_name}_state", auxiliary)
        setattr(trainer, f"_eval_{aux_name}_graphdef", "eval-aux")
    configured = trainer.configure_functions()
    state = trainer._eval_state_for_compiled_step(trainer.model_state)
    extra = trainer._eval_shared_fn_extra_args if aux_name else ()
    assert configured.sharded_evaluation_step_function(state, {}, *extra) == (
        "eval-policy",
        ("eval-aux",) if aux_name else (),
    )
    assert trainer.model_state.graphdef == "training"
    if aux_name:
        # Runtime variants must use the latest live auxiliary state, not a
        # snapshot captured before self-distillation initialization or sync.
        updated = getattr(trainer, f"{aux_name}_state").replace(opt_state="new-buffers")
        setattr(trainer, f"{aux_name}_state", updated)
        assert trainer._eval_shared_fn_extra_args[0].opt_state == "new-buffers"


@pytest.mark.parametrize("readonly", [False, True])
def test_generated_labels_do_not_corrupt_or_require_writable_tokens(monkeypatch, readonly):
    monkeypatch.setattr(gkd.jnp, "asarray", np.asarray)
    trainer = object.__new__(gkd.GKDTrainer)
    trainer.arguments = SimpleNamespace(max_length=4)
    trainer.pad_token_id = 0
    sequences = np.array([[2, 3, 5, 0]], dtype=np.int32)
    sequences.flags.writeable = not readonly
    original = {"input_ids": np.array([[2, 3]], dtype=np.int32), "labels": np.array([[-100, -100]], dtype=np.int32)}
    result = trainer._build_batch_from_sequences(original, sequences, prompt_seq_len=2)
    np.testing.assert_array_equal(result["input_ids"], [[2, 3, 5, 0]])
    np.testing.assert_array_equal(sequences, [[2, 3, 5, 0]])
    np.testing.assert_array_equal(result["labels"], [[-100, -100, 5, -100]])
    np.testing.assert_array_equal(result["completion_mask"], [[0, 0, 1, 0]])


def test_sequence_kd_generator_uses_teacher_contract(monkeypatch):
    monkeypatch.setattr(gkd, "replicated_named_sharding", lambda mesh: "replicated")
    monkeypatch.setattr(gkd, "compile_trainer_step", compile_contract)
    outputs = SimpleNamespace(sequences=[[2, 3, 5]])
    model = SimpleNamespace(mesh=SimpleNamespace(__enter__=lambda: None), generate=lambda **kwargs: outputs)

    class Mesh:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    model.mesh = Mesh()
    trainer = object.__new__(gkd.GKDTrainer)
    trainer.teacher_state = State("teacher-architecture", model=model)
    trainer.generation_config = SimpleNamespace()
    trainer._prepare_generation_config = lambda config: config
    generated = trainer._create_teacher_generate_function(shard_inputs=False)
    assert generated(trainer.teacher_state, [2, 3], [1, 1]) == ([[2, 3, 5]], [2, 3], [1, 1])
    with pytest.raises(ValueError, match="optimizer metadata"):
        generated(trainer.teacher_state.replace(tx="adam", opt_state="slots"), [2, 3], [1, 1])


def test_sequence_kd_does_not_silently_fall_back_on_generation_failure(monkeypatch):
    trainer = object.__new__(gkd.GKDTrainer)
    trainer._extract_prompts = lambda batch: (np.array([[2, 3]]), np.ones((1, 2)), np.array([2]))

    def fail(*args):
        raise ValueError("bad generation state")

    trainer.gkd_teacher_generate_function = fail
    with pytest.raises(RuntimeError, match="Sequence-KD teacher generation failed") as error:
        trainer._apply_generation(State("teacher"), {}, source="teacher")
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("ld_alpha", [0.0, 0.4])
def test_reference_precomputation_preserves_ld_scoring(monkeypatch, ld_alpha):
    # Stand-in token scorer with unequal completion lengths. Verify the public
    # precomputation entrypoint returns tail-weighted scores, not plain sums.
    def scorer(model, *, batch, ld_alpha=None, **kwargs):
        alpha = 1.0 if ld_alpha is None else ld_alpha
        return {"chosen_logps": -1.0, "rejected_logps": -1.0 - alpha}

    monkeypatch.setattr(dpo, "concatenated_forward", scorer)
    trainer = object.__new__(dpo.DPOTrainer)
    trainer.reference_state = State("reference", model=SimpleNamespace(eval=lambda: None))
    trainer.arguments = Arguments(ld_alpha=ld_alpha)
    trainer.padding_value = 0
    assert trainer.compute_reference_log_probs({}) == (-1.0, -1.0 - ld_alpha)


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("is_train", [False, True])
def test_moving_reference_discards_both_cached_score_conventions(monkeypatch, sync, is_train):
    # Host preprocessing feeds both ordinary and scheduled compiled steps.
    monkeypatch.setattr(Trainer, "_preprocess_batch_input", lambda self, state, batch, is_train: (dict(batch), {}))
    trainer = object.__new__(dpo.DPOTrainer)
    trainer.arguments = SimpleNamespace(sync_ref_model=sync)
    batch = {
        "input_ids": [[2, 3]],
        "ref_chosen_logps": [1],
        "ref_rejected_logps": [2],
        "reference_chosen_log_probs": [3],
        "reference_rejected_log_probs": [4],
    }
    prepared, _ = trainer._preprocess_batch_input(None, batch, is_train)
    assert prepared == ({"input_ids": [[2, 3]]} if sync else batch)
    assert len(batch) == 5
