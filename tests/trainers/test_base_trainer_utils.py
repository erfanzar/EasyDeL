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

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from easydel.infra.errors import EasyDeLPreemptionSignal
from easydel.trainers.base_trainer import BaseTrainer, GenerationResults
from easydel.trainers.proximal_policy_optimization_trainer.modeling_value_head import CausalLMWithValueHead
from easydel.trainers.trainer.trainer import Trainer


class _PreviewTrainer(BaseTrainer):
    def train(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def configure_functions(self):
        raise NotImplementedError

    def create_collect_function(self, max_sequence_length, truncation_mode: str = "keep_end"):
        raise NotImplementedError

    def create_grain_collect_function(self, max_sequence_length, truncation_mode: str = "keep_end"):
        raise NotImplementedError

    def create_tfds_collect_function(self, max_sequence_length, truncation_mode: str = "keep_end"):
        raise NotImplementedError

    def _run_training_loop(self, *args, **kwargs):
        raise NotImplementedError

    def _run_evaluation(self, *args, **kwargs):
        raise NotImplementedError

    def _train_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def _eval_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def _execute_train_step(self, *args, **kwargs):
        raise NotImplementedError

    def _execute_eval_step(self, *args, **kwargs):
        raise NotImplementedError

    def _finalize_training(self, *args, **kwargs):
        raise NotImplementedError


class _NoopTimer:
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def __init__(self):
        self.logged: list[str] = []

    def __call__(self, _name: str):
        return self._Ctx()

    def log(self, name: str):
        self.logged.append(name)


class _MeshCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


_CountState = namedtuple("_CountState", ["count", "payload"])


class _StateStub:
    def __init__(self, *, opt_state, tx, step=0):
        self.opt_state = opt_state
        self.tx = tx
        self.step = step
        self.shardings = "state-shardings"
        self.init_tx_calls: list[object] = []
        self.replace_calls: list[dict[str, object]] = []
        self.shard_state_calls: list[dict[str, object]] = []

    def init_tx(self, tx):
        self.init_tx_calls.append(tx)
        self.tx = tx
        self.opt_state = (
            _CountState(count=jnp.asarray(0, dtype=jnp.int32), payload={"initialized": True}),
            {"count": jnp.asarray(0, dtype=jnp.int32)},
        )
        return self

    def replace(self, **kwargs):
        self.replace_calls.append(dict(kwargs))
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def shard_state(self, *, partition_rules, mesh):
        self.shard_state_calls.append({"partition_rules": partition_rules, "mesh": mesh})
        return self


class _ModelStub:
    def __init__(self, rules):
        self.mesh = _MeshCtx()
        self._rules = rules

    def _get_partition_rules(self, _):
        return self._rules


class _CheckpointerStub:
    def __init__(self, *, should_save: bool = True):
        self.should_save = should_save
        self._last_save_step = 0
        self._last_save_time = None
        self._dt_now_injection = lambda: "synced-now"
        self.calls: list[dict[str, object]] = []

    def on_step(self, *, mesh, pytree, step, force=False, true_callbacks):
        self.calls.append(
            {
                "mesh": mesh,
                "pytree": pytree,
                "step": step,
                "force": force,
            }
        )
        if not self.should_save:
            return
        for callback in true_callbacks:
            callback(f"run-{step}", mesh, {"step": step, "is_temporary": False})


def test_configure_state_initializes_tx_then_shards_via_state_api():
    trainer = object.__new__(_PreviewTrainer)
    trainer.timer = _NoopTimer()
    trainer.arguments = SimpleNamespace(init_tx=True)
    trainer._resumed_from_checkpoint = False
    trainer.tx = "tx-object"
    trainer.model_state = _StateStub(opt_state=None, tx=None, step=0)
    trainer._model = _ModelStub(rules=((".*", "pspec"),))

    BaseTrainer._configure_state(trainer)

    assert trainer.model_state.init_tx_calls == ["tx-object"]
    assert trainer.model_state.shard_state_calls == [{"partition_rules": ((".*", "pspec"),), "mesh": trainer.model.mesh}]
    assert trainer.state_shardings == "state-shardings"
    assert trainer.timer.logged == ["configure sharded state"]


def test_configure_state_resume_keeps_step_and_sets_runtime_tx_before_sharding():
    trainer = object.__new__(_PreviewTrainer)
    trainer.timer = _NoopTimer()
    trainer.arguments = SimpleNamespace(init_tx=True)
    trainer._resumed_from_checkpoint = True
    trainer.tx = "new-tx"
    trainer.model_state = _StateStub(opt_state={"loaded": True}, tx="old-tx", step=17)
    trainer._model = _ModelStub(rules=((".*", "pspec"),))

    BaseTrainer._configure_state(trainer)

    assert trainer.model_state.init_tx_calls == []
    assert {"tx": "new-tx"} in trainer.model_state.replace_calls
    assert {"step": 17} in trainer.model_state.replace_calls
    assert trainer.model_state.shard_state_calls == [{"partition_rules": ((".*", "pspec"),), "mesh": trainer.model.mesh}]
    assert trainer.state_shardings == "state-shardings"


def test_apply_step_start_point_initializes_fresh_state_step():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(step_start_point=13)
    trainer._resumed_from_checkpoint = False
    trainer.model_state = _StateStub(opt_state=None, tx=None, step=0)

    BaseTrainer._apply_step_start_point(trainer)

    assert int(trainer.model_state.step) == 13
    assert any("step" in call and int(call["step"]) == 13 for call in trainer.model_state.replace_calls)


def test_apply_step_start_point_normalizes_matching_step_to_jax_array():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(step_start_point=13)
    trainer._resumed_from_checkpoint = False
    trainer.model_state = _StateStub(opt_state=None, tx=None, step=13)

    BaseTrainer._apply_step_start_point(trainer)

    assert isinstance(trainer.model_state.step, jax.Array)
    assert int(trainer.model_state.step) == 13
    assert any("step" in call and int(call["step"]) == 13 for call in trainer.model_state.replace_calls)


def test_apply_runtime_model_config_overrides_sets_lmhead_chunksize_on_all_states():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(lmhead_chunksize=96)
    trainer.model_state = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(lmhead_chunksize=None)))
    trainer.reference_state = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(lmhead_chunksize=None)))
    trainer.ref_state = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(lmhead_chunksize=None)))
    trainer.teacher_state = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(lmhead_chunksize=None)))

    BaseTrainer._apply_runtime_model_config_overrides(trainer)

    assert trainer.model_state.model.config.lmhead_chunksize == 96
    assert trainer.reference_state.model.config.lmhead_chunksize == 96
    assert trainer.ref_state.model.config.lmhead_chunksize == 96
    assert trainer.teacher_state.model.config.lmhead_chunksize == 96


def test_apply_runtime_model_config_overrides_preserves_config_when_arg_missing():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(lmhead_chunksize=None)
    trainer.model_state = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(lmhead_chunksize=64)))

    BaseTrainer._apply_runtime_model_config_overrides(trainer)

    assert trainer.model_state.model.config.lmhead_chunksize == 64


def test_apply_runtime_model_config_overrides_on_late_bound_reference_state_assignment():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(lmhead_chunksize=160)
    trainer.ref_state = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(lmhead_chunksize=None)))

    assert trainer.ref_state.model.config.lmhead_chunksize == 160


def test_memory_optimization_hints_are_trainer_specific():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(
        trainer_prefix="GRPO",
        logprob_vocab_chunk_size=None,
        lmhead_chunksize=None,
        ref_logps_chunk_size=None,
        completion_chunk_size=None,
        max_loss_completion_tokens=None,
        total_batch_size=8,
        gradient_accumulation_steps=1,
        max_prompt_length=1024,
        max_completion_length=2048,
        num_return_sequences=4,
    )

    hint_text = BaseTrainer._format_memory_optimization_hints(trainer)

    assert hint_text is not None
    assert "trainer `GRPO`" in hint_text
    assert "`logprob_vocab_chunk_size` (current: disabled): enable it" in hint_text
    assert "`ref_logps_chunk_size` (current: disabled): enable it" in hint_text
    assert "`completion_chunk_size` (current: disabled): enable it" in hint_text
    assert "`max_loss_completion_tokens` (current: disabled): set it" in hint_text
    assert "`num_return_sequences` (current: 4): lower it" in hint_text


def test_is_memory_oom_exception_uses_jax_runtime_error_type():
    assert BaseTrainer._is_memory_oom_exception(jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: CompileTimeHbmOom"))
    assert not BaseTrainer._is_memory_oom_exception(jax.errors.JaxRuntimeError("INVALID_ARGUMENT: shape mismatch"))


def test_execute_train_step_annotates_memory_oom_with_supported_knobs():
    trainer = object.__new__(Trainer)
    trainer.pruning_module = None
    trainer.arguments = SimpleNamespace(
        trainer_prefix="DPO",
        logprob_vocab_chunk_size=None,
        lmhead_chunksize=None,
        total_batch_size=4,
        gradient_accumulation_steps=1,
        max_length=4096,
        max_prompt_length=2048,
        max_completion_length=2048,
    )
    trainer._train_shared_fn_extra_args = ()
    trainer._train_shared_fn_static_args = ()
    trainer.sharded_training_step_function = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("RESOURCE_EXHAUSTED: CompileTimeHbmOom")
    )

    state, metrics, run_exception = Trainer._execute_train_step(
        trainer,
        state=SimpleNamespace(),
        batch={"input_ids": np.arange(4, dtype=np.int32)},
    )

    assert state is not None
    assert metrics is not None
    assert isinstance(run_exception, RuntimeError)
    assert "CompileTimeHbmOom" in str(run_exception)
    assert "Memory optimization techniques available for trainer `DPO`" in str(run_exception)
    assert "`logprob_vocab_chunk_size` (current: disabled): enable it" in str(run_exception)


def test_apply_step_start_point_ignores_nonzero_state_step():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(step_start_point=13)
    trainer._resumed_from_checkpoint = False
    trainer.model_state = _StateStub(opt_state=None, tx=None, step=4)

    with patch("easydel.trainers.base_trainer.logger.warning") as warning:
        BaseTrainer._apply_step_start_point(trainer)

    assert int(trainer.model_state.step) == 4
    assert not any("step" in call for call in trainer.model_state.replace_calls)
    warning.assert_called_once()


def test_apply_step_start_point_overrides_resumed_checkpoint_step():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(step_start_point=13)
    trainer._resumed_from_checkpoint = True
    trainer.model_state = _StateStub(opt_state={"loaded": True}, tx="old-tx", step=4)

    BaseTrainer._apply_step_start_point(trainer)

    assert isinstance(trainer.model_state.step, jax.Array)
    assert int(trainer.model_state.step) == 13
    assert any("step" in call and int(call["step"]) == 13 for call in trainer.model_state.replace_calls)


def test_configure_state_seeds_opt_state_counts_from_step_start_point():
    trainer = object.__new__(_PreviewTrainer)
    trainer.timer = _NoopTimer()
    trainer.arguments = SimpleNamespace(init_tx=True, step_start_point=13)
    trainer._resumed_from_checkpoint = False
    trainer.tx = "tx-object"
    trainer.model_state = _StateStub(opt_state=None, tx=None, step=jnp.asarray(13, dtype=jnp.int32))
    trainer._model = _ModelStub(rules=((".*", "pspec"),))

    BaseTrainer._configure_state(trainer)

    assert trainer.model_state.init_tx_calls == ["tx-object"]
    assert int(trainer.model_state.opt_state[0].count) == 13
    assert int(trainer.model_state.opt_state[1]["count"]) == 13


def test_configure_state_resume_seeds_opt_state_counts_from_step_start_point():
    trainer = object.__new__(_PreviewTrainer)
    trainer.timer = _NoopTimer()
    trainer.arguments = SimpleNamespace(init_tx=True, step_start_point=13)
    trainer._resumed_from_checkpoint = True
    trainer.tx = "new-tx"
    trainer.model_state = _StateStub(
        opt_state=(
            _CountState(count=jnp.asarray(4, dtype=jnp.int32), payload={"loaded": True}),
            {"count": jnp.asarray(4, dtype=jnp.int32)},
        ),
        tx="old-tx",
        step=jnp.asarray(13, dtype=jnp.int32),
    )
    trainer._model = _ModelStub(rules=((".*", "pspec"),))

    BaseTrainer._configure_state(trainer)

    assert trainer.model_state.init_tx_calls == []
    assert {"tx": "new-tx"} in trainer.model_state.replace_calls
    assert int(trainer.model_state.opt_state[0].count) == 13
    assert int(trainer.model_state.opt_state[1]["count"]) == 13


def test_save_checkpoint_for_step_updates_checkpointer_bookkeeping_for_callback_saves():
    trainer = object.__new__(_PreviewTrainer)
    trainer._model = SimpleNamespace(mesh="mesh")
    trainer.arguments = SimpleNamespace(_get_save_directory=lambda: Path("/tmp/easydel-checkpoints"))
    trainer.checkpointer = _CheckpointerStub()

    saved_paths: list[str] = []
    cleanup_calls: list[str] = []

    def _save_state(*, state, save_directory):
        saved_paths.append(save_directory)
        return save_directory

    trainer._save_state = _save_state
    trainer._cleanup_old_checkpoints = lambda: cleanup_calls.append("called")

    saved = BaseTrainer._save_checkpoint_for_step(trainer, state="state", step=7)

    assert saved == "/tmp/easydel-checkpoints/run-7"
    assert saved_paths == ["/tmp/easydel-checkpoints/run-7"]
    assert cleanup_calls == ["called"]
    assert trainer.checkpointer._last_save_step == 7
    assert trainer.checkpointer._last_save_time == "synced-now"
    assert trainer.checkpointer.calls == [
        {
            "mesh": "mesh",
            "pytree": None,
            "step": 7,
            "force": False,
        }
    ]


def test_fast_forward_batches_replays_iterator_with_wraparound_semantics():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(ids_to_pop_from_dataset=None)
    dataloader = [{"value": 0}, {"value": 1}, {"value": 2}]

    data_iter = BaseTrainer._fast_forward_batches(trainer, iter(dataloader), dataloader, 4)
    batch, _ = BaseTrainer._get_next_batch(trainer, data_iter, dataloader)

    assert batch == {"value": 1}


def test_fast_forward_batches_raises_on_empty_dataloader():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(ids_to_pop_from_dataset=None)

    with pytest.raises(RuntimeError, match="empty"):
        BaseTrainer._fast_forward_batches(trainer, iter(()), (), 1)


def test_get_next_batch_raises_runtime_error_for_empty_dataloader():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(ids_to_pop_from_dataset=None)

    with pytest.raises(RuntimeError, match="empty"):
        BaseTrainer._get_next_batch(trainer, iter(()), ())


def test_trainer_save_state_forwards_standard_save_kwargs(tmp_path):
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(
        save_arguments=lambda path: None,
        save_optimizer_state=False,
        _get_save_directory_milestone=lambda step, create=True: tmp_path / f"run-{step}",
    )
    trainer._model = SimpleNamespace(param_dtype=jnp.bfloat16)
    trainer._save_readme = lambda directory: None
    save_calls: list[dict[str, object]] = []

    class _State:
        def save_state(self, **kwargs):
            save_calls.append(dict(kwargs))

    saved = BaseTrainer._save_state(
        trainer,
        state=_State(),
        save_directory=tmp_path / "explicit",
    )

    assert saved == str(tmp_path / "explicit")
    assert len(save_calls) == 1
    assert set(save_calls[0]) == {"save_directory", "float_dtype", "save_optimizer"}
    assert str(save_calls[0]["save_directory"]) == str(tmp_path / "explicit")
    assert save_calls[0]["float_dtype"] is jnp.bfloat16
    assert save_calls[0]["save_optimizer"] is False


def test_get_current_step_uses_state_step_without_step_start_point_offset():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(step_start_point=9)

    assert BaseTrainer._get_current_step(trainer, SimpleNamespace(step=7)) == 7


def test_save_tpu_preemption_checkpoint_uses_standard_checkpoint_naming():
    trainer = object.__new__(_PreviewTrainer)
    trainer._preemption_checkpoint_path = None
    save_calls: list[dict[str, object]] = []

    def _save_checkpoint_for_step(*, state, step, force=False):
        save_calls.append({"state": state, "step": step, "force": force})
        return f"/tmp/easydel-checkpoints/run-{step}"

    trainer._save_checkpoint_for_step = _save_checkpoint_for_step

    with patch("jax.experimental.multihost_utils.sync_global_devices") as sync_global_devices:
        saved = BaseTrainer._save_tpu_preemption_checkpoint(trainer, state="state", step=9)

    assert saved == "/tmp/easydel-checkpoints/run-9"
    assert trainer._preemption_checkpoint_path == "/tmp/easydel-checkpoints/run-9"
    assert save_calls == [{"state": "state", "step": 9, "force": True}]
    assert [call.args[0] for call in sync_global_devices.call_args_list] == [
        "tpu-preemption-save-9-start",
        "tpu-preemption-save-9-done",
    ]


def test_should_save_tpu_preemption_checkpoint_uses_jax_sync_point():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(save_tpu_preemption_checkpoints=True)
    trainer._tpu_preemption_sync_available = None

    with (
        patch("jax.default_backend", return_value="tpu"),
        patch("jax.experimental.multihost_utils.reached_preemption_sync_point", return_value=True) as sync_point,
    ):
        assert BaseTrainer._should_save_tpu_preemption_checkpoint(trainer, step=11) is True

    sync_point.assert_called_once_with(11)
    assert trainer._tpu_preemption_sync_available is True


def test_should_save_tpu_preemption_checkpoint_disables_on_missing_jax_service():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(save_tpu_preemption_checkpoints=True)
    trainer._tpu_preemption_sync_available = None

    with (
        patch("jax.default_backend", return_value="tpu"),
        patch(
            "jax.experimental.multihost_utils.reached_preemption_sync_point",
            side_effect=RuntimeError("preemption sync manager missing"),
        ),
        patch("easydel.trainers.base_trainer.logger.warning_once") as warning_once,
    ):
        assert BaseTrainer._should_save_tpu_preemption_checkpoint(trainer, step=11) is False

    warning_once.assert_called_once()
    assert trainer._tpu_preemption_sync_available is False


def test_prepare_training_output_prefers_preemption_checkpoint_without_extra_last_save():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(do_last_save=True, save_directory="/tmp/easydel-checkpoints")
    trainer._model = SimpleNamespace(mesh="mesh")
    trainer._preemption_checkpoint_path = "/tmp/easydel-checkpoints/run-9"

    def _unexpected_save(*args, **kwargs):
        raise AssertionError("unexpected final save")

    trainer._save_checkpoint_for_step = _unexpected_save

    output = BaseTrainer._prepare_training_output(
        trainer,
        state=SimpleNamespace(step=9),
        run_exception=EasyDeLPreemptionSignal("TPU preemption checkpoint saved"),
    )

    assert output.checkpoint_path == "/tmp/easydel-checkpoints/run-9"
    assert output.last_save_file_name == "/tmp/easydel-checkpoints/run-9"


def test_prepare_training_output_treats_plain_stop_iteration_as_runtime_error():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(do_last_save=False, save_directory="/tmp/easydel-checkpoints")
    trainer._model = SimpleNamespace(mesh="mesh")
    trainer._preemption_checkpoint_path = None

    with pytest.raises(RuntimeError, match="unexpected iterator exhaustion"):
        BaseTrainer._prepare_training_output(
            trainer,
            state=SimpleNamespace(step=0),
            run_exception=StopIteration("unexpected iterator exhaustion"),
        )


def test_normalize_prompts_plain_string():
    normalized = BaseTrainer._normalize_esurge_prompts("hello", apply_chat_template=False)
    assert normalized == ["hello"]


def test_normalize_prompts_chat_wrapping():
    normalized = BaseTrainer._normalize_esurge_prompts("hello", apply_chat_template=True)
    assert len(normalized) == 1
    convo = normalized[0]
    assert isinstance(convo, list)
    assert convo[0]["role"] == "user"
    assert convo[0]["content"] == "hello"


def test_normalize_prompts_double_wrapped_chat_passes_through():
    chat = [[{"role": "user", "content": "hi"}]]
    normalized = BaseTrainer._normalize_esurge_prompts(chat, apply_chat_template=False)
    assert normalized == chat


def test_normalize_prompts_list_of_strings():
    prompts = ["first", "second"]
    normalized = BaseTrainer._normalize_esurge_prompts(prompts, apply_chat_template=False)
    assert normalized == prompts


def test_prepare_generation_input_accepts_chat_prompt_field():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(max_length=8, generation_dataset_prompt_field="generation_prompt")

    class _Processor:
        def __init__(self):
            self.padding_side = "right"
            self.calls: list[tuple[object, dict[str, object]]] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if kwargs.get("tokenize", False):
                return {
                    "input_ids": np.asarray([[101, 102, 103]], dtype=np.int32),
                    "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int32),
                }
            return "<chat prompt>"

    processor = _Processor()
    trainer.processing_class = processor
    trainer._batch_decode_tokens = lambda token_ids: ["decoded"]

    prompt = {
        "generation_prompt": [
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "solve x"},
        ]
    }

    prepared = BaseTrainer._prepare_generation_input(trainer, prompt)

    assert prepared is not None
    np.testing.assert_array_equal(np.asarray(prepared["input_ids"]), np.asarray([[101, 102, 103]], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(prepared["attention_mask"]), np.asarray([[1, 1, 1]], dtype=np.int32))
    assert prepared["prompt_text"] == "<chat prompt>"
    assert len(processor.calls) == 2
    assert processor.calls[0][0] == prompt["generation_prompt"]
    assert processor.calls[0][1]["tokenize"] is True
    assert processor.calls[1][1]["tokenize"] is False


def test_prepare_generation_input_passes_dataset_tools_to_chat_template():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(max_length=8, generation_dataset_prompt_field="generation_prompt")

    class _Processor:
        def __init__(self):
            self.padding_side = "right"
            self.calls: list[tuple[object, object, dict[str, object]]] = []

        def apply_chat_template(self, messages, tools=None, **kwargs):
            self.calls.append((messages, tools, kwargs))
            if kwargs.get("tokenize", False):
                return {
                    "input_ids": np.asarray([[101, 102, 103]], dtype=np.int32),
                    "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int32),
                }
            return "<chat prompt with tools>"

    processor = _Processor()
    trainer.processing_class = processor
    trainer._batch_decode_tokens = lambda token_ids: ["decoded"]

    tools = [
        {
            "name": "lookup_weather",
            "description": "Get the weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }
    ]
    prompt = {
        "generation_prompt": [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "What is the weather in Paris?"},
        ],
        "tools": tools,
    }

    prepared = BaseTrainer._prepare_generation_input(trainer, prompt)

    assert prepared is not None
    np.testing.assert_array_equal(np.asarray(prepared["input_ids"]), np.asarray([[101, 102, 103]], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(prepared["attention_mask"]), np.asarray([[1, 1, 1]], dtype=np.int32))
    assert prepared["prompt_text"] == "<chat prompt with tools>"
    assert len(processor.calls) == 2
    assert processor.calls[0][0] == prompt["generation_prompt"]
    assert processor.calls[0][1] == tools
    assert processor.calls[0][2]["tokenize"] is True
    assert processor.calls[1][1] == tools
    assert processor.calls[1][2]["tokenize"] is False


def test_maybe_generate_batches_prompts_and_maps_multiple_return_sequences():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(
        generation_interval=2,
        generation_num_return_sequences=2,
        use_esurge_generation=True,
        generation_shard_inputs=False,
        use_wandb=False,
        can_log_metrics=False,
        generation_log_to_wandb=False,
        generation_preview_print=False,
    )
    trainer._pad_token_id = 0
    trainer.latest_generation_samples = []

    prompts = ["prompt-1", "prompt-2"]
    prepared = {
        "prompt-1": {
            "input_ids": jnp.asarray([[11, 12, 13]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
            "prompt_text": "Prompt 1",
        },
        "prompt-2": {
            "input_ids": jnp.asarray([[21, 22, 23]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
            "prompt_text": "Prompt 2",
        },
    }
    trainer._collect_generation_prompts = lambda: prompts
    trainer._prepare_generation_input = lambda prompt: prepared[prompt]
    trainer._batch_decode_tokens = lambda token_ids: ["decoded-1", "decoded-2"]

    generate_calls: list[dict[str, object]] = []

    def fake_generate_unified(
        *,
        input_ids,
        attention_mask,
        state,
        use_esurge,
        apply_chat_template,
        shard_inputs,
        all_gather,
    ):
        generate_calls.append(
            {
                "input_ids": np.asarray(input_ids),
                "attention_mask": np.asarray(attention_mask),
                "state": state,
                "use_esurge": use_esurge,
                "apply_chat_template": apply_chat_template,
                "shard_inputs": shard_inputs,
                "all_gather": all_gather,
            }
        )
        return GenerationResults(
            generation_results=[
                "prompt-1-completion-0",
                "prompt-1-completion-1",
                "prompt-2-completion-0",
                "prompt-2-completion-1",
            ],
            prompt_ids=jnp.asarray(input_ids, dtype=jnp.int32),
            prompt_mask=jnp.asarray(attention_mask, dtype=jnp.int32),
            sequences=jnp.zeros((4, 8), dtype=jnp.int32),
            completion_ids=jnp.zeros((4, 5), dtype=jnp.int32),
            completion_mask=jnp.ones((4, 5), dtype=jnp.int32),
            decoded_prompts=["Prompt 1", "Prompt 2"],
            completion_prompts=["Prompt 1", "Prompt 1", "Prompt 2", "Prompt 2"],
            reasoning=["r1-0", None, "r2-0", "r2-1"],
            tool_calls=[[{"name": "lookup-0"}], None, [{"name": "lookup-2"}], []],
        )

    trainer.generate_unified = fake_generate_unified

    class _Model:
        def __init__(self):
            self.pause_calls = 0

        def pause_esurge(self, **kwargs):
            self.pause_calls += 1

    model = _Model()
    state = SimpleNamespace(model=model)

    trainer.maybe_generate(state=state, step=2)

    assert len(generate_calls) == 1
    call = generate_calls[0]
    assert call["use_esurge"] is True
    assert call["apply_chat_template"] is False
    assert call["all_gather"] is False
    assert call["shard_inputs"] is False
    np.testing.assert_array_equal(call["input_ids"], np.asarray([[11, 12, 13], [21, 22, 23]], dtype=np.int32))
    np.testing.assert_array_equal(call["attention_mask"], np.asarray([[1, 1, 1], [1, 1, 1]], dtype=np.int32))

    assert trainer.latest_generation_samples == [
        {
            "prompt": "Prompt 1",
            "completions": ["prompt-1-completion-0", "prompt-1-completion-1"],
            "reasoning": ["r1-0", "No reasoning content ..."],
            "tool_calls": ["[{'name': 'lookup-0'}]", "No tools were called ..."],
            "step": 2,
        },
        {
            "prompt": "Prompt 2",
            "completions": ["prompt-2-completion-0", "prompt-2-completion-1"],
            "reasoning": ["r2-0", "r2-1"],
            "tool_calls": ["[{'name': 'lookup-2'}]", "No tools were called ..."],
            "step": 2,
        },
    ]
    assert model.pause_calls == 0


def test_maybe_generate_skips_malformed_prompt_when_batching():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(
        generation_interval=1,
        generation_num_return_sequences=1,
        use_esurge_generation=True,
        generation_shard_inputs=False,
        use_wandb=False,
        can_log_metrics=False,
        generation_log_to_wandb=False,
        generation_preview_print=False,
    )
    trainer._pad_token_id = 0
    trainer.latest_generation_samples = []

    prompts = ["valid", "invalid"]
    prepared = {
        "valid": {
            "input_ids": jnp.asarray([[11, 12, 13]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
            "prompt_text": "Valid Prompt",
        },
        "invalid": {
            "input_ids": jnp.asarray([[21, 22, 23]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1]], dtype=jnp.int32),
            "prompt_text": "Invalid Prompt",
        },
    }
    trainer._collect_generation_prompts = lambda: prompts
    trainer._prepare_generation_input = lambda prompt: prepared[prompt]
    trainer._batch_decode_tokens = lambda token_ids: ["decoded-valid"]

    generate_calls: list[dict[str, object]] = []

    def fake_generate_unified(
        *,
        input_ids,
        attention_mask,
        state,
        use_esurge,
        apply_chat_template,
        shard_inputs,
        all_gather,
    ):
        generate_calls.append(
            {
                "input_ids": np.asarray(input_ids),
                "attention_mask": np.asarray(attention_mask),
                "state": state,
                "use_esurge": use_esurge,
                "apply_chat_template": apply_chat_template,
                "shard_inputs": shard_inputs,
                "all_gather": all_gather,
            }
        )
        return GenerationResults(
            generation_results=["valid-completion"],
            prompt_ids=jnp.asarray(input_ids, dtype=jnp.int32),
            prompt_mask=jnp.asarray(attention_mask, dtype=jnp.int32),
            sequences=jnp.zeros((1, 8), dtype=jnp.int32),
            completion_ids=jnp.zeros((1, 5), dtype=jnp.int32),
            completion_mask=jnp.ones((1, 5), dtype=jnp.int32),
            decoded_prompts=["Valid Prompt"],
            completion_prompts=["Valid Prompt"],
        )

    trainer.generate_unified = fake_generate_unified

    class _Model:
        def pause_esurge(self, **kwargs):
            del kwargs

    state = SimpleNamespace(model=_Model())
    trainer.maybe_generate(state=state, step=1)

    assert len(generate_calls) == 1
    call = generate_calls[0]
    np.testing.assert_array_equal(call["input_ids"], np.asarray([[11, 12, 13]], dtype=np.int32))
    np.testing.assert_array_equal(call["attention_mask"], np.asarray([[1, 1, 1]], dtype=np.int32))
    assert trainer.latest_generation_samples == [
        {
            "prompt": "Valid Prompt",
            "completions": ["valid-completion"],
            "reasoning": ["No reasoning content ..."],
            "tool_calls": ["No tools were called ..."],
            "step": 1,
        }
    ]


def test_maybe_generate_prefers_completion_aligned_text_field():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(
        generation_interval=1,
        generation_num_return_sequences=1,
        use_esurge_generation=True,
        generation_shard_inputs=False,
        use_wandb=False,
        can_log_metrics=False,
        generation_log_to_wandb=False,
        generation_preview_print=False,
    )
    trainer._pad_token_id = 0
    trainer.latest_generation_samples = []

    trainer._collect_generation_prompts = lambda: ["prompt-1"]
    trainer._prepare_generation_input = lambda prompt: {
        "input_ids": jnp.asarray([[11, 12, 13]], dtype=jnp.int32),
        "attention_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
        "prompt_text": "Prompt 1",
    }
    trainer._batch_decode_tokens = lambda token_ids: ["decoded"]

    def fake_generate_unified(
        *,
        input_ids,
        attention_mask,
        state,
        use_esurge,
        apply_chat_template,
        shard_inputs,
        all_gather,
    ):
        del state, use_esurge, apply_chat_template, shard_inputs, all_gather
        return GenerationResults(
            generation_results=["legacy-sequence-text"],
            prompt_ids=jnp.asarray(input_ids, dtype=jnp.int32),
            prompt_mask=jnp.asarray(attention_mask, dtype=jnp.int32),
            sequences=jnp.zeros((1, 8), dtype=jnp.int32),
            completion_ids=jnp.zeros((1, 5), dtype=jnp.int32),
            completion_mask=jnp.ones((1, 5), dtype=jnp.int32),
            decoded_prompts=["Prompt 1"],
            completion_prompts=["Prompt 1"],
            text=["parsed-completion"],
            reasoning=["step-by-step"],
            tool_calls=[[{"name": "lookup", "arguments": "{}"}]],
        )

    trainer.generate_unified = fake_generate_unified

    class _Model:
        def pause_esurge(self, **kwargs):
            del kwargs

    trainer.maybe_generate(state=SimpleNamespace(model=_Model()), step=1)

    assert trainer.latest_generation_samples == [
        {
            "prompt": "Prompt 1",
            "completions": ["parsed-completion"],
            "reasoning": ["step-by-step"],
            "tool_calls": ["[{'arguments': '{}', 'name': 'lookup'}]"],
            "step": 1,
        }
    ]


def test_generate_unified_esurge_releases_only_used_engine():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(
        generation_max_new_tokens=2,
        max_completion_length=None,
        generation_temperature=0.7,
        generation_top_p=0.95,
        generation_top_k=64,
        generation_num_return_sequences=1,
        esurge_hbm_utilization=None,
        esurge_max_num_seqs=None,
        esurge_min_input_pad=None,
        esurge_page_size=None,
        esurge_silent_mode=True,
        esurge_runner_verbose=True,
        esurge_max_num_batched_tokens=None,
        esurge_enable_prefix_caching=None,
        esurge_data_parallelism_axis=None,
        esurge_max_num_seq_buckets=None,
        total_batch_size=1,
        max_length=8,
        esurge_use_tqdm=False,
        use_esurge_generation=True,
    )
    trainer._pad_token_id = 0
    trainer.processing_class = "tok"

    class _Completion:
        def __init__(self, token_ids):
            self.token_ids = token_ids
            self.text = "completion-text"
            self.reasoning_content = "reasoning-text"
            self.tool_calls = [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
            self.raw_text = "<tool_call>completion-text</tool_call>"

    class _RequestOutput:
        def __init__(self):
            self.prompt_token_ids = [[11, 12]]
            self.outputs = [_Completion([13])]
            self.accumulated_text = "completion-text"
            self.raw_accumulated_text = "<tool_call>completion-text</tool_call>"
            self.prompt = "prompt-text"

    class _Engine:
        def __init__(self):
            self.pause_calls = 0
            self.release_calls: list[bool] = []

        def pause(self):
            self.pause_calls += 1

        def release_model_state(self, *, clear_compiled_cache: bool = False):
            self.release_calls.append(clear_compiled_cache)

    class _Model:
        def __init__(self, engine):
            self._engine = engine
            self.call_esurge_engine_kwargs = None

        def get_esurge(self, **kwargs):
            self.get_esurge_kwargs = kwargs
            return self._engine

        def _call_esurge_engine(self, engine, **kwargs):
            assert engine is self._engine
            self.call_esurge_engine_kwargs = kwargs
            return [_RequestOutput()]

    engine = _Engine()
    model = _Model(engine)
    state = SimpleNamespace(model=model)

    results = trainer.generate_unified(
        prompts=["prompt-text"],
        state=state,
        use_esurge=True,
        apply_chat_template=False,
        shard_inputs=False,
        all_gather=False,
    )

    assert model.call_esurge_engine_kwargs is not None
    assert model.get_esurge_kwargs["runner_verbose"] is True
    assert results.generation_results == ["completion-text"]
    assert results.text == ["completion-text"]
    assert results.reasoning == ["reasoning-text"]
    assert results.tool_calls == [[{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}]]
    assert results.raw_text == ["<tool_call>completion-text</tool_call>"]
    assert engine.pause_calls == 1
    assert engine.release_calls == [False]


def test_generate_unified_esurge_propagates_generation_penalties():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(
        generation_max_new_tokens=2,
        max_completion_length=None,
        generation_temperature=0.7,
        generation_top_p=0.95,
        generation_top_k=64,
        generation_presence_penalty=0.4,
        generation_frequency_penalty=0.2,
        generation_repetition_penalty=1.1,
        generation_num_return_sequences=1,
        esurge_hbm_utilization=None,
        esurge_max_num_seqs=None,
        esurge_min_input_pad=None,
        esurge_page_size=None,
        esurge_silent_mode=True,
        esurge_runner_verbose=False,
        esurge_max_num_batched_tokens=None,
        esurge_enable_prefix_caching=None,
        esurge_data_parallelism_axis=None,
        esurge_max_num_seq_buckets=None,
        total_batch_size=1,
        max_length=8,
        esurge_use_tqdm=False,
        use_esurge_generation=True,
    )
    trainer._pad_token_id = 0
    trainer.processing_class = "tok"

    class _Completion:
        def __init__(self, token_ids):
            self.token_ids = token_ids

    class _RequestOutput:
        def __init__(self):
            self.prompt_token_ids = [[11, 12]]
            self.outputs = [_Completion([13])]
            self.accumulated_text = "completion-text"
            self.prompt = "prompt-text"

    class _Engine:
        def __init__(self):
            self.pause_calls = 0
            self.release_calls: list[bool] = []

        def pause(self):
            self.pause_calls += 1

        def release_model_state(self, *, clear_compiled_cache: bool = False):
            self.release_calls.append(clear_compiled_cache)

    class _Model:
        def __init__(self, engine):
            self._engine = engine
            self.call_esurge_engine_kwargs = None

        def get_esurge(self, **kwargs):
            self.get_esurge_kwargs = kwargs
            return self._engine

        def _call_esurge_engine(self, engine, **kwargs):
            assert engine is self._engine
            self.call_esurge_engine_kwargs = kwargs
            return [_RequestOutput()]

    engine = _Engine()
    model = _Model(engine)
    state = SimpleNamespace(model=model)

    trainer.generate_unified(
        prompts=["prompt-text"],
        state=state,
        use_esurge=True,
        apply_chat_template=False,
        shard_inputs=False,
        all_gather=False,
        config_overrides={
            "presence_penalty": 0.6,
            "frequency_penalty": 0.3,
            "repetition_penalty": 1.4,
        },
    )

    sampling_params = model.call_esurge_engine_kwargs["sampling_params"]
    assert sampling_params.presence_penalty == pytest.approx(0.6)
    assert sampling_params.frequency_penalty == pytest.approx(0.3)
    assert sampling_params.repetition_penalty == pytest.approx(1.4)


def test_generate_unified_compiled_populates_completion_aligned_text_fields():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(
        generation_max_new_tokens=2,
        max_completion_length=None,
        generation_temperature=0.7,
        generation_top_p=0.95,
        generation_top_k=64,
        generation_presence_penalty=0.0,
        generation_frequency_penalty=0.0,
        generation_repetition_penalty=1.0,
        generation_num_return_sequences=1,
        use_esurge_generation=False,
    )
    trainer._pad_token_id = 0
    trainer.generate_function = None
    trainer.generate_function_with_model_kwargs = None
    trainer.model_state = SimpleNamespace(model=SimpleNamespace())
    trainer._model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=99))

    class _Processor:
        pad_token_id = 0
        eos_token_id = 99

        @staticmethod
        def decode(seq, skip_special_tokens=True):
            values = [int(token) for token in np.asarray(seq).tolist()]
            if skip_special_tokens:
                values = [token for token in values if token not in (0, 99)]
            return "|".join(str(token) for token in values)

    trainer.processing_class = _Processor()

    def fake_generate_aio(*, input_ids, attention_mask, **kwargs):
        del kwargs
        return (
            jnp.asarray([[11, 12, 0, 99, 21]], dtype=jnp.int32),
            jnp.asarray(input_ids, dtype=jnp.int32),
            jnp.asarray(attention_mask, dtype=jnp.int32),
        )

    trainer.generate_aio = fake_generate_aio

    results = trainer.generate_unified(
        input_ids=jnp.asarray([[11, 12, 0]], dtype=jnp.int32),
        attention_mask=jnp.asarray([[1, 1, 0]], dtype=jnp.int32),
        state=trainer.model_state,
        use_esurge=False,
        shard_inputs=False,
        release_runtime_after_generation=False,
        all_gather=False,
    )

    assert results.generation_results == ["11|12|21"]
    assert results.text == ["21"]
    assert results.reasoning == [None]
    assert results.tool_calls == [None]
    assert results.raw_text == ["99|21"]


def test_maybe_generate_falls_back_to_per_prompt_after_batch_failure():
    trainer = object.__new__(_PreviewTrainer)
    trainer.arguments = SimpleNamespace(
        generation_interval=1,
        generation_num_return_sequences=1,
        use_esurge_generation=True,
        generation_shard_inputs=False,
        use_wandb=False,
        can_log_metrics=False,
        generation_log_to_wandb=False,
        generation_preview_print=False,
    )
    trainer._pad_token_id = 0
    trainer.latest_generation_samples = []

    prompts = ["good", "bad"]
    prepared = {
        "good": {
            "input_ids": jnp.asarray([[11, 12, 13]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
            "prompt_text": "Good Prompt",
        },
        "bad": {
            "input_ids": jnp.asarray([[21, 22, 23]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.int32),
            "prompt_text": "Bad Prompt",
        },
    }
    trainer._collect_generation_prompts = lambda: prompts
    trainer._prepare_generation_input = lambda prompt: prepared[prompt]
    trainer._batch_decode_tokens = lambda token_ids: ["decoded-good"]

    call_shapes: list[tuple[int, int]] = []

    def fake_generate_unified(
        *,
        input_ids,
        attention_mask,
        state,
        use_esurge,
        apply_chat_template,
        shard_inputs,
        all_gather,
    ):
        del state, use_esurge, apply_chat_template, shard_inputs, all_gather
        input_np = np.asarray(input_ids)
        mask_np = np.asarray(attention_mask)
        call_shapes.append((int(input_np.shape[0]), int(input_np.shape[1])))

        if input_np.shape[0] > 1:
            raise RuntimeError("batched generation failed")
        if int(input_np[0, 0]) == 21:
            raise RuntimeError("single prompt failed")

        return GenerationResults(
            generation_results=["good-completion"],
            prompt_ids=jnp.asarray(input_np, dtype=jnp.int32),
            prompt_mask=jnp.asarray(mask_np, dtype=jnp.int32),
            sequences=jnp.zeros((1, 8), dtype=jnp.int32),
            completion_ids=jnp.zeros((1, 5), dtype=jnp.int32),
            completion_mask=jnp.ones((1, 5), dtype=jnp.int32),
            decoded_prompts=["Good Prompt"],
            completion_prompts=["Good Prompt"],
        )

    trainer.generate_unified = fake_generate_unified

    class _Model:
        def pause_esurge(self, **kwargs):
            del kwargs

    state = SimpleNamespace(model=_Model())
    trainer.maybe_generate(state=state, step=1)

    assert call_shapes == [(2, 3), (1, 3), (1, 3)]
    assert trainer.latest_generation_samples == [
        {
            "prompt": "Good Prompt",
            "completions": ["good-completion"],
            "reasoning": ["No reasoning content ..."],
            "tool_calls": ["No tools were called ..."],
            "step": 1,
        }
    ]


def test_value_head_wrapper_delegates_call_esurge_engine():
    calls: dict[str, object] = {}

    class _BaseModel:
        def _call_esurge_engine(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return ["ok"]

    wrapper_like = SimpleNamespace(model=_BaseModel())
    result = CausalLMWithValueHead._call_esurge_engine(wrapper_like, "engine", prompts=["hello"])

    assert result == ["ok"]
    assert calls["args"] == ("engine",)
    assert calls["kwargs"] == {"prompts": ["hello"]}


def test_maybe_benchmark_runs_named_benchmark_suite_and_logs_metrics(monkeypatch):
    trainer = object.__new__(_PreviewTrainer)
    logged_metrics: list[tuple[dict[str, float], int]] = []
    wandb_calls: dict[str, object] = {}
    trainer.arguments = SimpleNamespace(
        benchmark_interval=2,
        benchmarks=[
            {
                "name": "code_suite",
                "tasks": ["humaneval"],
                "enable_thinking": True,
                "max_new_tokens": 256,
            }
        ],
        esurge_hbm_utilization=None,
        esurge_max_num_seqs=8,
        esurge_min_input_pad=None,
        esurge_page_size=None,
        esurge_silent_mode=True,
        esurge_runner_verbose=False,
        esurge_max_num_batched_tokens=None,
        esurge_enable_prefix_caching=None,
        esurge_data_parallelism_axis=None,
        esurge_max_num_seq_buckets=None,
        eval_batch_size=4,
        total_batch_size=4,
        max_length=1024,
        use_wandb=True,
        can_log_metrics=True,
        log_metrics=lambda metrics, step: logged_metrics.append((metrics, step)),
    )
    trainer.processing_class = object()
    trainer.latest_benchmark_results = {}
    trainer.benchmark_log_table = None

    run_calls: list[dict[str, object]] = []

    def _fake_run_lm_eval_with_esurge(**kwargs):
        run_calls.append(kwargs)
        return {"results": {"humaneval": {"pass@1,create_test": 0.5}}}

    monkeypatch.setattr("easydel.trainers.base_trainer.run_lm_eval_with_esurge", _fake_run_lm_eval_with_esurge)

    class _DummyTable:
        def __init__(self, *, columns, log_mode):
            wandb_calls["columns"] = columns
            wandb_calls["log_mode"] = log_mode
            self.rows: list[tuple[object, ...]] = []

        def add_data(self, *row):
            self.rows.append(row)

    class _DummyWandb:
        Table = _DummyTable

        @staticmethod
        def log(payload, step):
            wandb_calls["payload"] = payload
            wandb_calls["step"] = step

    monkeypatch.setattr("easydel.trainers.base_trainer.wandb", _DummyWandb)

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(granted_freq_max_position_embedding=4096)
            self.pause_calls: list[dict[str, object]] = []

        def get_esurge(self, **kwargs):
            self.last_esurge_kwargs = kwargs
            return "engine"

        def pause_esurge(self, **kwargs):
            self.pause_calls.append(kwargs)

    state = SimpleNamespace(model=_Model())

    trainer.maybe_benchmark(state=state, step=2)

    assert len(run_calls) == 1
    assert run_calls[0]["tasks"] == ["humaneval"]
    assert run_calls[0]["eval_config"]["enable_thinking"] is True
    assert run_calls[0]["eval_config"]["max_new_tokens"] == 256
    assert run_calls[0]["stop_engine"] is False
    assert trainer.latest_benchmark_results["code_suite"]["results"]["humaneval"]["pass@1,create_test"] == 0.5
    assert logged_metrics == [({"benchmark/code_suite/humaneval/pass@1,create_test": 0.5}, 2)]
    assert state.model.pause_calls == [{"release_model_state": True, "clear_compiled_cache": False}]
    assert wandb_calls["columns"] == ["step", "benchmark", "task", "metric", "value"]
    assert wandb_calls["log_mode"] == "INCREMENTAL"
    assert trainer.benchmark_log_table.rows == [(2, "code_suite", "humaneval", "pass@1,create_test", 0.5)]
    assert wandb_calls["payload"] == {"benchmark_results": trainer.benchmark_log_table}
    assert wandb_calls["step"] == 2
