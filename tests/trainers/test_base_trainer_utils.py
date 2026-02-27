from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from easydel.trainers.base_trainer import BaseTrainer, GenerationResults
from easydel.trainers.proximal_policy_optimization_trainer.modeling_value_head import CausalLMWithValueHead


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
            "step": 2,
        },
        {
            "prompt": "Prompt 2",
            "completions": ["prompt-2-completion-0", "prompt-2-completion-1"],
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
    assert engine.pause_calls == 1
    assert engine.release_calls == [False]


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
