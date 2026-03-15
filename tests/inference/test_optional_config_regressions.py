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

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from flax import nnx as nn

import easydel as ed
from easydel.inference.esurge.config import CacheConfig, Config, SchedulerConfig
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.mixins.parsing import EngineParsingMixin
from easydel.inference.esurge.mixins.utils import EngineUtilsMixin
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.evaluations.esurge_eval import eSurgeLMEvalAdapter
from easydel.inference.reasoning.parsers.qwen3_reasoning_parser import Qwen3ReasoningParser
from easydel.inference.sampling_params import SamplingParams
from easydel.infra.utils import AttnMaskType


class _DummyProcessor:
    """Minimal processor stub for lm-eval adapter tests."""

    pad_token_id = None
    eos_token_id = 1
    padding_side = "right"

    def __init__(self) -> None:
        self.last_chat_template_kwargs: dict[str, object] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        """Return a deterministic chat template rendering."""
        self.last_chat_template_kwargs = {
            "messages": list(messages),
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **kwargs,
        }
        return "dummy-template"

    def encode(self, string: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in string]

    def decode(
        self,
        tokens,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens, spaces_between_special_tokens
        return "".join(chr(int(token)) for token in tokens)


class _BoundaryAwareProcessor(_DummyProcessor):
    """Processor stub with boundary-sensitive tokenization for scoring tests."""

    bos_token_id = 11
    eos_token_id = 12

    def encode(self, string: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        table = {
            "hello": [1],
            "world": [2],
            " world": [3],
            "hello world": [1, 3],
            "<bos>answer": [11, 21],
            "answer": [21],
        }
        if string in table:
            return list(table[string])
        return super().encode(string, add_special_tokens=False)


class _ThinkingProcessor(_DummyProcessor):
    """Processor stub that records `enable_thinking` overrides."""

    def __init__(self) -> None:
        self.last_enable_thinking: bool | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        **kwargs,
    ) -> str:
        """Capture the thinking flag while returning a stable template."""
        self.last_enable_thinking = enable_thinking
        self.last_chat_template_kwargs = {
            "messages": list(messages),
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
            **kwargs,
        }
        return "thinking-template"


class _EmptyReasoningScaffoldProcessor(_DummyProcessor):
    """Processor stub that injects an empty reasoning scaffold when continuing."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        enable_thinking: bool = True,
        **kwargs,
    ) -> str:
        self.last_chat_template_kwargs = {
            "messages": list(messages),
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
            "enable_thinking": enable_thinking,
            **kwargs,
        }
        return "<|im_start|>assistant\n<think>\n\n</think>\n\nprefix"


class _DummySurge:
    """Minimal eSurge stub that satisfies adapter initialization."""

    max_num_seqs = 2

    class _Runner:
        model = None

    runner = _Runner()
    _scheduler_running = True

    @property
    def think_start_token(self) -> str | None:
        return "<think>"

    @property
    def think_end_token(self) -> str | None:
        return "</think>"


class _NoReasoningMetadataSurge(_DummySurge):
    """Minimal surge stub without exposed reasoning boundary metadata."""

    @property
    def think_start_token(self) -> str | None:
        return None

    @property
    def think_end_token(self) -> str | None:
        return None


class _DummyTokenizer:
    """Minimal tokenizer stub for reasoning parser tests."""

    def get_vocab(self) -> dict[str, int]:
        return {}


class _ParsingHarness(EngineParsingMixin, EngineUtilsMixin):
    """Small helper exposing parser/stop-policy logic for unit tests."""

    ignore_stop_strings_in_reasoning = False
    extra_stops = None
    _generation_config_dict = None
    _primary_eos_token_id = None
    _sampling_params_callback = None


class _FakeGenerateOutput:
    """Small response stub matching the `eSurge.generate` output API."""

    def __init__(self, request_id: str, text: str, accumulated_text: str | None = None):
        self.request_id = request_id
        self._text = text
        self.accumulated_text = text if accumulated_text is None else accumulated_text

    def get_text(self) -> str:
        """Return the generated text payload."""
        return self._text


class _RecordingSurge(_DummySurge):
    """eSurge test double that records generate calls."""

    def __init__(self, text: str = "pass") -> None:
        self.calls: list[dict[str, object]] = []
        self.text = text

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        request_id: list[str],
        use_tqdm: bool = False,
    ) -> list[_FakeGenerateOutput]:
        """Record generation inputs and return fixed-text outputs."""
        self.calls.append(
            {
                "prompts": list(prompts),
                "sampling_params": sampling_params,
                "request_id": list(request_id),
                "use_tqdm": use_tqdm,
            }
        )
        return [_FakeGenerateOutput(rid, self.text) for rid in request_id]


def test_esurge_eval_rejects_non_iterable_math_hint_input(monkeypatch):
    monkeypatch.setattr(eSurgeLMEvalAdapter, "_setup", lambda self: None)

    with pytest.raises(TypeError, match="math_answer_task_hints"):
        eSurgeLMEvalAdapter(
            _DummySurge(),
            _DummyProcessor(),
            math_answer_task_hints=False,
        )


def test_esurge_eval_prefix_token_id_prefers_bos(monkeypatch):
    monkeypatch.setattr(eSurgeLMEvalAdapter, "_setup", lambda self: None)

    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        _BoundaryAwareProcessor(),
    )

    assert adapter.prefix_token_id == 11


def test_esurge_eval_loglikelihood_matches_template_lm_pair_encoding(monkeypatch):
    monkeypatch.setattr(eSurgeLMEvalAdapter, "_setup", lambda self: None)

    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        _BoundaryAwareProcessor(),
        batch_size=8,
    )

    captured: list[tuple[list[list[int]], list[list[int]]]] = []

    def _capture(ctx_ids, cont_ids):
        captured.append((ctx_ids, cont_ids))
        return [(0.0, True)] * len(ctx_ids)

    monkeypatch.setattr(adapter, "_loglikelihood_token_ids", _capture)

    instances = [
        SimpleNamespace(args=("hello ", "world")),
        SimpleNamespace(args=("", "<bos>answer")),
    ]

    result = adapter.loglikelihood(instances)

    assert result == [(0.0, True), (0.0, True)]
    assert captured == [
        (
            [[1], [11]],
            [[3], [21]],
        )
    ]


def test_esurge_eval_accepts_single_string_math_hint(monkeypatch):
    monkeypatch.setattr(eSurgeLMEvalAdapter, "_setup", lambda self: None)

    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        _DummyProcessor(),
        math_answer_task_hints="gsm8k",
    )

    assert adapter.math_answer_task_hints == ("gsm8k",)


def test_esurge_eval_honors_request_generation_kwargs():
    """Project lm-eval request kwargs into the effective sampling parameters."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(_DummySurge(), _DummyProcessor())
    adapter.surge = surge

    instances = [
        SimpleNamespace(
            arguments=[
                "def foo(x):\n",
                {
                    "until": ["\ndef"],
                    "max_gen_toks": 17,
                    "do_sample": False,
                    "top_p": 0.91,
                    "top_k": 7,
                    "min_p": 0.2,
                    "include_stop_str_in_output": True,
                },
            ],
            task_name="humaneval",
        )
    ]

    outputs = adapter.generate_until(instances)

    assert outputs == ["pass"]
    assert len(surge.calls) == 1
    sampling_params = surge.calls[0]["sampling_params"]
    assert sampling_params.max_tokens == 17
    assert sampling_params.temperature == 0.0
    assert sampling_params.top_k == 0
    assert sampling_params.min_p == 0.0
    assert sampling_params.include_stop_str_in_output is True
    assert "\ndef" in sampling_params.stop


def test_esurge_eval_can_force_global_max_new_tokens():
    """When enabled, the adapter should ignore task-local max token overrides."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        _DummyProcessor(),
        max_new_tokens=9,
        hard_max_new_tokens=True,
    )
    adapter.surge = surge

    instances = [
        SimpleNamespace(
            arguments=[
                "def foo(x):\n",
                {
                    "until": ["\ndef"],
                    "max_gen_toks": 17,
                    "max_tokens": 21,
                },
            ],
            task_name="humaneval",
        )
    ]

    outputs = adapter.generate_until(instances)

    assert outputs == ["pass"]
    assert len(surge.calls) == 1
    sampling_params = surge.calls[0]["sampling_params"]
    assert sampling_params.max_tokens == 9


def test_esurge_eval_can_ignore_benchmark_eos_flags():
    """Eval can drop task-provided stop strings and trust engine EOS only."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        _DummyProcessor(),
        ignore_benchmark_eos_flags=True,
    )
    adapter.surge = surge

    instances = [
        SimpleNamespace(
            arguments=[
                "def foo(x):\n",
                {
                    "until": ["\ndef", "\nif"],
                    "max_gen_toks": 17,
                },
            ],
            task_name="humaneval",
        )
    ]

    outputs = adapter.generate_until(instances)

    assert outputs == ["pass"]
    assert len(surge.calls) == 1
    sampling_params = surge.calls[0]["sampling_params"]
    assert sampling_params.stop == []


def test_esurge_eval_prefers_accumulated_text_when_completion_text_is_stale():
    """Treat RequestOutput.accumulated_text as authoritative when available."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(_DummySurge(), _DummyProcessor())
    adapter.surge = surge

    def _generate_with_stale_completion(
        prompts: list[str],
        sampling_params: SamplingParams,
        request_id: list[str],
        use_tqdm: bool = False,
    ) -> list[_FakeGenerateOutput]:
        del prompts, sampling_params, use_tqdm
        return [_FakeGenerateOutput(rid, text="", accumulated_text="def answer():\n    return 42") for rid in request_id]

    surge.generate = _generate_with_stale_completion

    outputs = adapter.generate_until(
        [
            SimpleNamespace(arguments=["prompt", {"until": []}], task_name="task"),
        ]
    )

    assert outputs == ["def answer():\n    return 42"]


def test_esurge_eval_splits_batches_when_request_sampling_kwargs_differ():
    """Separate prompts whose normalized sampling parameters do not match."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(_DummySurge(), _DummyProcessor())
    adapter.surge = surge

    instances = [
        SimpleNamespace(arguments=["prompt-a", {"until": [], "temperature": 0.7, "top_k": 4}], task_name="task-a"),
        SimpleNamespace(arguments=["prompt-b", {"until": [], "temperature": 0.7, "top_k": 8}], task_name="task-b"),
    ]

    outputs = adapter.generate_until(instances)

    assert outputs == ["pass", "pass"]
    assert len(surge.calls) == 2
    assert {call["sampling_params"].top_k for call in surge.calls} == {4, 8}


def test_esurge_eval_keeps_greedy_until_deterministic_even_with_request_sampling_overrides():
    """Ignore per-request sampling overrides for greedy decoding paths."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(_DummySurge(), _DummyProcessor())
    adapter.surge = surge

    instances = [
        SimpleNamespace(
            arguments=["prompt", {"until": [], "temperature": 0.7, "top_p": 0.4, "top_k": 11}],
            task_name="task",
        )
    ]

    outputs = adapter.greedy_until(instances)

    assert outputs == ["pass"]
    assert len(surge.calls) == 1
    sampling_params = surge.calls[0]["sampling_params"]
    assert sampling_params.temperature == 0.0
    assert sampling_params.top_p == 1.0
    assert sampling_params.top_k == 0


def test_esurge_eval_chat_template_disables_tokenizer_thinking():
    """Tokenizer chat rendering should suppress model-side thinking when supported."""
    processor = _ThinkingProcessor()
    adapter = eSurgeLMEvalAdapter(_DummySurge(), processor)

    rendered = adapter.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
    )

    assert rendered == "thinking-template"
    assert processor.last_enable_thinking is False


def test_esurge_eval_chat_template_can_enable_tokenizer_thinking():
    """Eval chat rendering should allow explicit thinking enablement."""
    processor = _ThinkingProcessor()
    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        processor,
        enable_thinking=True,
    )

    rendered = adapter.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
    )

    assert rendered == "thinking-template"
    assert processor.last_enable_thinking is True


def test_esurge_eval_chat_template_forwards_chat_template_args():
    """Additional chat-template kwargs should be forwarded to the tokenizer."""
    processor = _ThinkingProcessor()
    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        processor,
        chat_template_args={"foo": "bar"},
    )

    rendered = adapter.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
    )

    assert rendered == "thinking-template"
    assert processor.last_chat_template_kwargs["foo"] == "bar"


def test_esurge_eval_chat_template_strips_empty_reasoning_scaffold_when_thinking_disabled():
    """Rendered prompts should drop inert empty reasoning blocks for non-thinking evals."""
    processor = _EmptyReasoningScaffoldProcessor()
    adapter = eSurgeLMEvalAdapter(_DummySurge(), processor, enable_thinking=False)

    rendered = adapter.apply_chat_template(
        [{"role": "assistant", "content": "prefix"}],
        add_generation_prompt=False,
    )

    assert rendered == "<|im_start|>assistant\nprefix"


def test_esurge_eval_chat_template_strips_literal_empty_reasoning_scaffold_without_metadata():
    """Prompt cleanup should still remove the common literal scaffold without parser metadata."""
    processor = _EmptyReasoningScaffoldProcessor()
    adapter = eSurgeLMEvalAdapter(_NoReasoningMetadataSurge(), processor, enable_thinking=False)

    rendered = adapter.apply_chat_template(
        [{"role": "assistant", "content": "prefix"}],
        add_generation_prompt=False,
    )

    assert rendered == "<|im_start|>assistant\nprefix"


def test_esurge_eval_uses_vllm_style_detokenization_defaults_for_generation():
    """Eval generations should disable spaces_between_special_tokens by default."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(_DummySurge(), _DummyProcessor())
    adapter.surge = surge

    outputs = adapter.generate_until(
        [
            SimpleNamespace(arguments=["prompt", {"until": []}], task_name="task"),
        ]
    )

    assert outputs == ["pass"]
    sampling_params = surge.calls[0]["sampling_params"]
    assert sampling_params.skip_special_tokens is False
    assert sampling_params.spaces_between_special_tokens is False


def test_esurge_eval_can_strip_reasoning_prefix_with_think_end_token():
    """Visible eval text should be recoverable from raw outputs containing thinking blocks."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        _DummyProcessor(),
        think_end_token="</think>",
    )
    adapter.surge = surge

    def _generate_reasoning_prefixed(
        prompts: list[str],
        sampling_params: SamplingParams,
        request_id: list[str],
        use_tqdm: bool = False,
    ) -> list[_FakeGenerateOutput]:
        del prompts, sampling_params, use_tqdm
        return [_FakeGenerateOutput(rid, text="<think>plan</think>def answer():\n    return 42") for rid in request_id]

    surge.generate = _generate_reasoning_prefixed

    outputs = adapter.generate_until(
        [
            SimpleNamespace(arguments=["prompt", {"until": []}], task_name="task"),
        ]
    )

    assert outputs == ["def answer():\n    return 42"]


def test_esurge_eval_preserves_indentation_after_think_end_token():
    """Reasoning cleanup should not strip visible indentation from code completions."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        _DummyProcessor(),
        think_end_token="</think>",
    )
    adapter.surge = surge

    def _generate_indented_completion(
        prompts: list[str],
        sampling_params: SamplingParams,
        request_id: list[str],
        use_tqdm: bool = False,
    ) -> list[_FakeGenerateOutput]:
        del prompts, sampling_params, use_tqdm
        return [_FakeGenerateOutput(rid, text="<think>plan</think>\n    return 42") for rid in request_id]

    surge.generate = _generate_indented_completion

    outputs = adapter.generate_until(
        [
            SimpleNamespace(arguments=["prompt", {"until": []}], task_name="task"),
        ]
    )

    assert outputs == ["\n    return 42"]


def test_esurge_eval_pretruncates_generation_prompts_like_vllm():
    """Prompt text should be truncated before dispatch so eval matches lm-eval backends more closely."""
    surge = _RecordingSurge()
    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        _DummyProcessor(),
        max_length=6,
        max_new_tokens=2,
    )
    adapter.surge = surge

    outputs = adapter.generate_until(
        [
            SimpleNamespace(arguments=["abcdef", {"until": [], "max_gen_toks": 2}], task_name="task"),
        ]
    )

    assert outputs == ["pass"]
    assert surge.calls[0]["prompts"] == ["cdef"]


def test_engine_default_can_enable_reasoning_aware_stop_matching():
    """Engine defaults should propagate parser-aware stop matching to requests."""
    harness = _ParsingHarness()

    prepared = harness._prepare_sampling_params_for_request(
        SamplingParams(stop=["\nif"]),
        request_id="req-stop-aware",
        prompt="prompt",
    )

    assert prepared.ignore_stop_strings_in_reasoning is True


def test_esurge_exposes_reasoning_boundary_token_properties():
    """Engine metadata should expose the active parser's think delimiters."""
    engine = object.__new__(ed.eSurge)
    engine._reasoning_parser_class = Qwen3ReasoningParser
    engine.tokenizer = _DummyTokenizer()
    engine._scheduler_running = False
    engine._monitoring_initialized = False
    engine._profiling_active = False
    engine.runner = SimpleNamespace(shutdown=lambda: None)

    assert engine.think_start_token == "<think>"
    assert engine.think_end_token == "</think>"


def test_stop_strings_can_ignore_matches_inside_reasoning():
    """Parser-aware stop matching should only inspect visible content."""
    harness = _ParsingHarness()
    raw_output = "<think>plan\nif guard</think>def answer():\n    return 1\nif __name__ == '__main__':\n    pass"
    request_data = {
        "sampling_params": SamplingParams(
            stop=["\nif"],
            ignore_stop_strings_in_reasoning=True,
        ),
        "reasoning_parser_instance": Qwen3ReasoningParser(_DummyTokenizer()),
        "tool_parser_instance": None,
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
        "accumulated_reasoning": "",
        "accumulated_content": "",
        "decoder_visible_text": "",
    }

    parsed, visible_text, visible_delta, stop_hit, stop_reason = harness._parse_with_stop_string_policy(
        request_data,
        accumulated_text=raw_output,
        delta_text=raw_output,
        token_ids=[],
        finished=True,
    )

    assert stop_hit is True
    assert stop_reason == "\nif"
    assert parsed["accumulated_reasoning"] == "plan\nif guard"
    assert parsed["accumulated_content"] == "def answer():\n    return 1"
    assert visible_text == "def answer():\n    return 1"
    assert visible_delta == "def answer():\n    return 1"


@pytest.mark.parametrize("method_name", ["generate_until", "greedy_until"])
def test_esurge_eval_preserves_math_normalization_for_generation_methods(method_name: str):
    """Keep math answer post-processing active for both generation entry points."""
    surge = _RecordingSurge("The answer is 42.")
    adapter = eSurgeLMEvalAdapter(_DummySurge(), _DummyProcessor())
    adapter.surge = surge

    instances = [
        SimpleNamespace(
            arguments=["Solve 40+2", {"until": []}],
            task_name="gsm8k",
        )
    ]

    outputs = getattr(adapter, method_name)(instances)

    assert outputs == ["The answer is 42.\n#### 42"]


def test_scheduler_falls_back_to_model_len_when_batch_token_limit_is_none():
    config = Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=None,
            max_model_len=128,
            token_safety_margin=None,
        ),
        cache_config=CacheConfig(num_pages=16, page_size=8, enable_prefix_caching=False),
    )
    kv_cache_config = CacheGroupsConfig(
        num_pages=16,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=8,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=None,
            )
        ],
    )

    scheduler = Scheduler(config=config, kv_cache_config=kv_cache_config)

    assert scheduler.max_num_scheduled_tokens == 128
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 0


def test_scheduler_aborts_empty_prompt_request_instead_of_requeueing():
    config = Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_model_len=128,
            token_safety_margin=None,
        ),
        cache_config=CacheConfig(num_pages=16, page_size=8, enable_prefix_caching=False),
    )
    kv_cache_config = CacheGroupsConfig(
        num_pages=16,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=8,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=None,
            )
        ],
    )

    scheduler = Scheduler(config=config, kv_cache_config=kv_cache_config)
    request = EngineRequest(
        request_id="req-empty-prompt",
        prompt_token_ids=[],
        sampling_params=SamplingParams(max_tokens=8),
        eos_token_id=1,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert "req-empty-prompt" in output.finished_req_ids
    assert request.status == EngineRequestStatus.FINISHED_ABORTED
    assert scheduler.get_num_unfinished_requests() == 0
    assert "req-empty-prompt" not in scheduler.requests


def test_linear_attention_mask_type_is_accepted():
    assert AttnMaskType.from_hf("linear_attention") == AttnMaskType.FULL


def test_create_mesh_normalizes_multi_device_axis_dims_on_single_device(monkeypatch):
    captured = {}

    def _fake_create_mesh(*, axis_dims, **kwargs):
        captured["axis_dims"] = axis_dims
        return object()

    monkeypatch.setattr(jax, "device_count", lambda backend=None: 1)
    monkeypatch.setattr("eformer.escale.create_mesh", _fake_create_mesh)

    ed.EasyDeLBaseConfig.create_mesh(sharding_axis_dims=(1, 4, 1, -1, 1))

    assert captured["axis_dims"] == (1, 1, 1, -1, 1)


def _build_tiny_qwen35(attn_mechanism: str):
    """Construct a minimal text-only Qwen 3.5 model for attention tests."""
    config = ed.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        head_dim=16,
        attn_mechanism=attn_mechanism,
    )
    with config.mesh:
        model = ed.Qwen3_5ForCausalLM.lazy_init(
            config=config,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=nn.Rngs(0),
        )
    return model


def _build_tiny_qwen35_vlm(attn_mechanism: str):
    """Construct a minimal multimodal Qwen 3.5 model for attention tests."""
    text_config = ed.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        head_dim=16,
        attn_mechanism=attn_mechanism,
        rope_scaling={"rope_type": "default", "mrope_section": [8, 4, 4], "mrope_interleaved": True},
        partial_rotary_factor=0.5,
    )
    vision_config = ed.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=text_config.hidden_size,
        num_position_embeddings=128,
        deepstack_visual_indexes=[],
    )
    config = ed.Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=text_config.vocab_size - 1,
        video_token_id=text_config.vocab_size - 2,
        vision_start_token_id=text_config.vocab_size - 3,
        vision_end_token_id=text_config.vocab_size - 4,
    )
    with config.mesh:
        model = ed.Qwen3_5ForConditionalGeneration.lazy_init(
            config=config,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=nn.Rngs(0),
        )
    return model


def test_esurge_compatible_model_forces_ragged_on_tpu(monkeypatch):
    """TPU compatibility should switch text attention to the ragged backend."""
    model = _build_tiny_qwen35("unified_attention")
    import jax

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    compatible = model.esurge_compatible_model
    assert compatible.config.get_text_config().attn_mechanism == "ragged_page_attention_v3"


def test_esurge_compatible_model_forces_unified_on_gpu(monkeypatch):
    """GPU compatibility should switch text attention to unified attention."""
    model = _build_tiny_qwen35("sdpa")
    import jax

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    compatible = model.esurge_compatible_model
    assert compatible.config.get_text_config().attn_mechanism == "unified_attention"


def test_esurge_compatible_model_updates_vlm_text_attn_recursively():
    """Compatibility conversion should also rewrite nested VLM text configs."""
    import jax

    backend = jax.default_backend()
    if backend == "gpu":
        source_attn = "sdpa"
        expected_attn = "unified_attention"
    else:
        source_attn = "unified_attention"
        expected_attn = "ragged_page_attention_v3"

    model = _build_tiny_qwen35_vlm(source_attn)
    compatible = model.esurge_compatible_model
    assert compatible.config.get_text_config().attn_mechanism == expected_attn
