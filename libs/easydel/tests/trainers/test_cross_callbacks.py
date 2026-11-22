"""Host-only lifecycle coverage with real merge callbacks and fake checkpoint storage."""

from types import SimpleNamespace

import pytest
from easydel.infra.errors import EasyDeLTimerError
from easydel.trainers.merge_callback import MergeModelCallback
from easydel.trainers.trainer.trainer import Trainer

pytestmark = pytest.mark.cpu_ok


def make_trainer():
    trainer = object.__new__(Trainer)
    trainer.arguments = SimpleNamespace(do_last_save=False, merge_lora_before_save=False, save_directory=None)
    trainer._model = SimpleNamespace(mesh=None)
    trainer._preemption_checkpoint_path = None
    return trainer


@pytest.mark.parametrize("every_checkpoint", [True, False])
def test_registered_merge_callback_receives_real_save_and_successful_end(tmp_path, every_checkpoint):
    trainer = make_trainer()
    state = {"name": "policy"}
    published = []
    callback = MergeModelCallback(
        merge_at_every_checkpoint=every_checkpoint,
        target_state={"name": "target"},
        push_to_hub=True,
        publish_fn=published.append,
    )
    trainer.add_callback(callback)
    trainer.add_callback(callback)  # Identity registration cannot duplicate lifecycle calls.
    destination = tmp_path / "checkpoint"
    destination.mkdir()
    storage_events = []

    def save_state(*, state, save_directory, **kwargs):
        storage_events.append(state)
        return str(save_directory)

    def on_step(**kwargs):
        kwargs["true_callbacks"][0](str(destination), None, {"step": 7})

    trainer._save_state = save_state
    trainer._cleanup_old_checkpoints = lambda: None
    trainer.checkpointer = SimpleNamespace(on_step=on_step)
    result = trainer._save_checkpoint_for_step(state, step=7)
    assert result == str(destination)
    assert storage_events == [state]
    assert published == ([state] if every_checkpoint else [])
    output = trainer._prepare_training_output(state)
    assert output.state == state
    assert published == [state]
    trainer._prepare_training_output(state)
    assert published == [state]
    trainer.remove_callback(callback)
    assert trainer.callbacks == []


def test_no_save_and_failed_save_never_dispatch_events():
    trainer = make_trainer()
    calls = []
    trainer.add_callback(SimpleNamespace(on_save=lambda **kwargs: calls.append(kwargs)))
    trainer.checkpointer = SimpleNamespace(on_step=lambda **kwargs: None)
    assert trainer._save_checkpoint_for_step({}, step=1) is None
    assert calls == []

    def failed_save(**kwargs):
        raise OSError("storage failed")

    def on_step(**kwargs):
        kwargs["true_callbacks"][0]("/unused", None, {})

    trainer._save_state = failed_save
    trainer.checkpointer = SimpleNamespace(on_step=on_step)
    with pytest.raises(OSError, match="storage failed"):
        trainer._save_checkpoint_for_step({}, step=1)
    assert calls == []


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), EasyDeLTimerError("limit"), ValueError("failed")])
def test_unsuccessful_completion_does_not_emit_train_end(interruption):
    trainer = make_trainer()
    calls = []
    trainer.add_callback(SimpleNamespace(on_train_end=lambda **kwargs: calls.append(kwargs)))
    if isinstance(interruption, ValueError):
        with pytest.raises(RuntimeError, match="failed"):
            trainer._prepare_training_output({}, run_exception=interruption)
    else:
        trainer._prepare_training_output({}, run_exception=interruption)
    assert calls == []


def test_callback_signature_filtering_and_explicit_publishing_context(tmp_path):
    trainer = make_trainer()
    trainer.hub_model_id = "requested/repo"
    trainer.hub_kwargs = {"private": True}
    received = []

    class Callback:
        def on_save(self, state, policy_state, checkpoint_path, hub_model_id, hub_kwargs):
            received.append((state, policy_state, checkpoint_path, hub_model_id, hub_kwargs))

    trainer.add_callback(Callback)
    trainer._save_state = lambda **kwargs: str(kwargs["save_directory"])
    state = {"name": "state"}
    trainer.save_pretrained(state, str(tmp_path))
    assert received == [(state, state, str(tmp_path), "requested/repo", {"private": True})]


def test_callback_body_errors_propagate_without_retrying():
    trainer = make_trainer()
    calls = []

    def on_train_end(state):
        calls.append(state)
        raise TypeError("callback body")

    trainer.add_callback(SimpleNamespace(on_train_end=on_train_end))
    with pytest.raises(TypeError, match="callback body"):
        trainer._prepare_training_output({})
    assert calls == [{}]
    trainer._prepare_training_output({})
    assert calls == [{}]
