"""Embedding evaluation's static-state compile contract, without array computation."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from easydel.trainers.embedding_trainer import embedding_trainer as embedding

pytestmark = pytest.mark.cpu_ok


@dataclass(frozen=True)
class StaticState:
    graphdef: str

    def replace(self, **changes):
        return replace(self, **changes)


def compile_static_contract(fn, *, in_shardings, **kwargs):
    """Model the compiler's static pytree-metadata check without tracing arrays."""

    def compiled(state, batch):
        if state.graphdef != in_shardings[0].graphdef:
            raise ValueError("State and sharding tree have different static graph metadata")
        return fn(state, batch)

    return compiled


def evaluate_static_contract(state, batch, **objective):
    """Return the selected graph and objective; numerical coverage lives separately."""
    return state.graphdef, objective


@pytest.mark.parametrize("eval_graphdef", [None, "eval-override-graph"])
def test_embedding_configured_evaluator_accepts_runtime_state(monkeypatch, eval_graphdef):
    # Isolate compilation's static metadata contract, not JAX numerics or model
    # initialization. Use the real trainer configuration and base eval-state swap.
    monkeypatch.setattr(embedding, "replicated_named_sharding", lambda mesh: "replicated")
    monkeypatch.setattr(embedding, "compile_trainer_step", compile_static_contract)
    monkeypatch.setattr(embedding, "embedding_evaluation_step", evaluate_static_contract)
    trainer = object.__new__(embedding.EmbeddingTrainer)
    trainer._model = SimpleNamespace(mesh=None)
    trainer.arguments = SimpleNamespace(
        loss_type="mnrl",
        temperature=0.7,
        margin=0.3,
        normalize_embeddings=False,
        matryoshka_dims=[8, 16],
        step_partition_spec=None,
        gradient_accumulation_steps=1,
        mpmd_scheduler=None,
        ensure_checkpoint_path=lambda: None,
        get_streaming_checkpointer=lambda: None,
    )
    trainer._embedding_config = trainer.arguments
    trainer.scheduler = None
    trainer.state_shardings = StaticState("training-graph")
    trainer.model_state = StaticState("training-graph")
    trainer._eval_graphdef = eval_graphdef
    trainer._runtime_trace = lambda *args, **kwargs: None

    configured = trainer.configure_functions()
    runtime_state = trainer._eval_state_for_compiled_step(trainer.model_state)
    graphdef, objective = configured.sharded_evaluation_step_function(runtime_state, {})
    assert graphdef == (eval_graphdef or "training-graph")
    assert objective == {
        "loss_type": "mnrl",
        "temperature": 0.7,
        "margin": 0.3,
        "normalize": False,
        "matryoshka_dims": [8, 16],
        "partition_spec": None,
    }
    assert trainer.model_state.graphdef == "training-graph"
    assert trainer.state_shardings.graphdef == "training-graph"
    if eval_graphdef is not None:
        with pytest.raises(ValueError, match="static graph metadata"):
            configured.sharded_evaluation_step_function(trainer.model_state, {})
