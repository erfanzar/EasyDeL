"""Array-free regressions for scheduler lifetime, request outputs and streaming stops."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.config import eSurgeContextConfig, eSurgeRuntimeConfig
from easydel.inference.esurge.engine import bootstrap
from easydel.inference.esurge.engine.admission import RequestAdmission
from easydel.inference.esurge.engine.loop import EngineLoop
from easydel.inference.esurge.engine.registry import RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, FinishReason
from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.esurge.metrics import MetricsCollector
from easydel.inference.esurge.monitoring import PrometheusMetrics
from easydel.inference.esurge.scheduler.dp_scheduler import DPSchedulerClosed
from easydel.inference.sampling_params import SamplingParams
from easydel.modules.auto import AutoEasyDeLModelForCausalLM
from easydel.modules.auto.auto_modeling import PreTrainedLoading

from tests.inference.esurge.test_admission_scheduling_contracts import NoCacheMediaManager, Tokenizer
from tests.inference.esurge.test_backend_selection import HostModel
from tests.inference.esurge.test_engine_stub_behaviors import _ChatStub
from tests.inference.esurge.test_output_logprobs import pipeline_factory as pipeline_factory
from tests.inference.esurge.test_output_logprobs import submit

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("failure", ["closed", "interrupt", "setup"])
def test_thread_exit_clears_public_alive_and_wakes_waiters(overlap, failure):
    registry = RequestRegistry()
    event = threading.Event()
    registry.events["request"] = event
    errors = []

    def schedule():
        if failure == "interrupt":
            raise KeyboardInterrupt()
        raise DPSchedulerClosed("closed")

    def info(*args):
        if failure == "setup":
            raise ValueError("setup failed")

    loop = EngineLoop(
        scheduler=SimpleNamespace(schedule=schedule),
        runner=SimpleNamespace(execute_model_async=lambda *args: None),
        coordinator=SimpleNamespace(check_health=lambda: None, supports_overlap=True),
        scheduler_lock=registry.request_lock,
        emit_outputs=lambda outputs: None,
        drain_parser_stops=lambda: None,
        on_fatal=errors.append,
        on_stopped=registry.signal_all,
        heartbeat=lambda: None,
        handle_profiling_step=lambda: None,
        is_nonrecoverable=lambda exc: False,
        overlap_execution=overlap,
        async_scheduling=False,
        runner_verbose=False,
        info=info,
    )
    engine = eSurge.__new__(eSurge)
    engine._engine_loop = loop
    loop.start()
    loop.join(2)
    assert loop.thread is None
    assert not loop.running
    assert not engine.alive
    assert event.wait(0.1)
    assert bool(errors) is (failure == "setup")


def admission(registry, scheduled, *, max_len=64, context=None):
    return RequestAdmission(
        registry=registry,
        scheduler_submit=scheduled,
        tokenizer_client=SimpleNamespace(tokenize=lambda *args: [3]),
        tokenizer=Tokenizer(),
        context_config=eSurgeContextConfig.coerce_config(context),
        reserve_tokens=1,
        max_model_len=max_len,
        tool_parser_class=None,
        reasoning_parser_class=None,
        sampling_params_callback=lambda: None,
        generation_config_dict={},
        primary_eos_token_id=0,
        eos_token_ids=[0],
        extra_stops=[],
        ignore_stop_strings_in_reasoning=True,
        on_activity=lambda: None,
        info=lambda *args: None,
    )


@pytest.mark.parametrize("truncate", [False, True])
def test_public_admission_output_reports_execution_ids_not_retokenized_text(truncate):
    registry = RequestRegistry()
    scheduled = []
    boundary = admission(
        registry,
        scheduled.extend,
        max_len=4 if truncate else 64,
        context={"decode_truncated_prompt": False, "prefer_preserve_prompt": False},
    )
    supplied = [77, 88, 99, 111] if truncate else [77, 88]
    boundary.add_request("request", "non-round-trippable", SamplingParams(max_tokens=1), prompt_token_ids=supplied)
    output = registry.get_output("request")
    expected = [99, 111] if truncate else supplied
    assert scheduled[0].prompt_token_ids == expected
    assert output.prompt_token_ids == [expected]
    assert output.prompt == "non-round-trippable"
    assert supplied == ([77, 88, 99, 111] if truncate else [77, 88])


@pytest.mark.parametrize(
    "key,value",
    [
        ("attn_dtype", "float32"),
        ("kvdtype", "float32"),
        ("freq_max_position_embeddings", 123),
        ("mask_max_position_embeddings", 123),
    ],
)
def test_loader_forwards_supported_overrides_without_duplicate_keywords(monkeypatch, key, value):
    calls = []
    monkeypatch.setattr(
        AutoEasyDeLModelForCausalLM, "from_pretrained", lambda **kwargs: calls.append(kwargs) or HostModel()
    )
    loaded = bootstrap._load_model(
        "test",
        loading_kwargs=PreTrainedLoading.coerce_config(
            {"pretrained_model_name_or_path": "test", "config_kwargs": {key: value}}
        ),
        runtime_config=eSurgeRuntimeConfig.coerce_config({}),
        data_parallelism_axis="dp",
        dtype=None,
        decoding_engine="ring",
    )
    assert isinstance(loaded, HostModel)
    assert calls[0]["config_kwargs"][key] == value
    assert calls[0]["config_kwargs"]["attn_mechanism"] == "ring_paged"


def test_public_tool_chat_preserves_caller_bias_without_mutating_caller():
    engine = _ChatStub(Tokenizer())
    original = SamplingParams(logit_bias={7: -75.0})
    forwarded = engine.chat(
        [{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        sampling_params=original,
    )
    assert forwarded.logit_bias == {7: -75.0}
    assert forwarded.skip_special_tokens is False
    assert original.logit_bias == {7: -75.0}
    assert forwarded is not original
    forwarded.logit_bias[7] = 1.0
    assert original.logit_bias[7] == -75.0


@pytest.mark.parametrize(
    "reason,expected", [(FinishReason.STOP, "stop"), (FinishReason.LENGTH, "length"), (FinishReason.ABORT, "abort")]
)
@pytest.mark.parametrize("tokens", [[], [0], [65]])
def test_public_completion_preserves_finish_enum_including_falsy_stop(pipeline_factory, reason, expected, tokens):
    pipeline, _, output, _ = pipeline_factory()
    submit(pipeline, EngineCoreOutput("parent", tokens, finish_reason=reason))
    assert output.finished
    assert output.outputs[0].finish_reason == expected


@pytest.mark.parametrize("parsed_domain", [False, True])
@pytest.mark.parametrize(
    "text,stops,expected,terminal",
    [
        ("END", ["END"], "", False),
        ("safeABAB", ["ABAB", "BAB"], "safe", False),
        ("safeENx", ["END"], "safeENx", True),
        ("safeEN", ["END"], "safeEN", True),
    ],
)
def test_streamed_stop_prefixes_are_buffered_and_disambiguated(
    pipeline_factory, parsed_domain, text, stops, expected, terminal
):
    pipeline, registry, _output, _ = pipeline_factory(stop=stops)
    record = registry.get_record("parent")
    if parsed_domain:
        # A public parser interface with content-only output selects the same
        # parsed-content stop domain used when reasoning is enabled.
        class ContentParser:
            reasoning_parser = object()

            def process_delta(self, accumulated, delta, *args):
                return SimpleNamespace(
                    tool_calls=None,
                    to_dict=lambda: {
                        "accumulated_content": accumulated,
                        "delta_content": delta,
                        "accumulated_reasoning": "",
                        "delta_reasoning": None,
                        "tool_calls": None,
                        "delta_tool_calls": None,
                    },
                )

            def process_final(self, accumulated, *args):
                return self.process_delta(accumulated, accumulated)

        record.delegating_parser = ContentParser()
        record.sampling_params.ignore_stop_strings_in_reasoning = True
    engine = eSurge.__new__(eSurge)
    engine._registry = registry
    engine._request_lock = registry.request_lock
    engine._output_lock = registry.output_lock
    engine._request_events = registry.events
    engine._request_outputs = registry.outputs
    engine._max_request_outputs = None
    engine._scheduler_running = True
    engine._raise_if_scheduler_failed = lambda: None
    engine._recover_orphaned_request = lambda rid: False
    stream = engine._iter_request_snapshots("parent", stopped_message="stopped")
    deltas = []
    for character in text:
        submit(pipeline, EngineCoreOutput("parent", [ord(character)]))
        snapshot = next(stream)
        deltas.append(snapshot.delta_text)
        if snapshot.finished:
            break
    if terminal:
        submit(pipeline, EngineCoreOutput("parent", [], finish_reason=FinishReason.LENGTH))
        snapshot = next(stream)
        deltas.append(snapshot.delta_text)
    assert snapshot.finished
    assert snapshot.accumulated_text == expected
    assert "".join(deltas) == expected
    stream.close()


def test_metrics_cursor_survives_bounded_history_wraparound_without_recounting():
    collector = MetricsCollector(history_size=1)
    exporter = PrometheusMetrics(prefix="history_wrap_")
    try:
        for _ in range(101):
            for _ in range(2):
                collector.record_scheduler_metrics(num_waiting=0, num_running=1, num_scheduled_tokens=1, num_preempted=1)
            exporter.update_from_metrics_collector(collector)
            exporter.update_from_metrics_collector(collector)
        samples = {
            sample.name: sample.value
            for family in exporter.preempted_requests_total.collect()
            for sample in family.samples
        }
        assert samples["history_wrap_preempted_requests_total"] == 101
    finally:
        exporter.close()


class HostEngine(eSurge):
    """Real admission and consumption with an array-free scheduler boundary."""

    def __init__(self, cap):
        self._registry = RequestRegistry(max_outputs=cap)
        self._request_lock = self._registry.request_lock
        self._output_lock = self._registry.output_lock
        self._request_events = self._registry.events
        self._request_outputs = self._registry.outputs
        self._active_requests = self._registry.records
        self._output_event = self._registry.output_event
        self._scheduler_lock = threading.RLock()
        self._scheduler_running = True
        self._max_request_outputs = cap
        self._detokenizer_client = SimpleNamespace(reset=lambda rid: None)
        self._failed_detokenizer_resets = set()
        self._detokenizer_cleanup_threshold = 10
        self.tokenizer = Tokenizer()
        self.scheduler = SimpleNamespace(requests={}, running=[], waiting=[], finish_requests=self.finish_requests)
        self._admission = admission(self._registry, self.submit_requests)
        self._multimodal_manager = NoCacheMediaManager()
        self.ready = threading.Event()

    def submit_requests(self, requests):
        self.scheduler.requests.update({request.request_id: request for request in requests})

    def finish_requests(self, ids, status):
        for rid in ids:
            self.scheduler.requests.pop(rid, None)

    def _tokenize_prompt(self, request_id, prompt):
        return [77, 88]

    def _format_chat_prompt(self, messages, **kwargs):
        return "hello"

    def _build_tool_parser_request(self, **kwargs):
        return None

    def _ensure_scheduler_running(self, **kwargs):
        for output in self._request_outputs.values():
            output.accumulated_text = "x"
            output.delta_text = "x"
            output.num_generated_tokens = 1
            output.update_seq = 1
        self._registry.signal_all()
        self.ready.set()

    def _raise_if_scheduler_failed(self):
        pass

    def _recover_orphaned_request(self, rid):
        return False


@pytest.mark.parametrize("cap", [0, 1])
@pytest.mark.parametrize("mode", ["stream", "generate", "multimodal"])
def test_public_abort_hands_terminal_output_to_active_consumers_before_eviction(monkeypatch, cap, mode):
    monkeypatch.setattr(esurge_engine, "get_metrics_collector", lambda: None)
    monkeypatch.setattr(esurge_engine, "log_metrics_summary", lambda: None)
    engine = HostEngine(cap)
    if mode == "stream":
        stream = engine.stream("hello", request_id="request")
        assert next(stream).finished is False
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(engine.abort_request, "request").result(timeout=2)
        # Other completions may roll the history while this consumer is asleep.
        for rid in ("other-a", "other-b"):
            engine._registry.outputs[rid] = SimpleNamespace(finished=True)
            engine._registry.track_finished(rid)
        terminal = next(stream)
        stream.close()
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            if mode == "generate":
                future = executor.submit(engine.generate, "hello", request_id="request", use_tqdm=False)
            else:
                messages = [{"role": "user", "content": [{"type": "image", "image": object()}]}]
                future = executor.submit(engine.chat, messages, request_id="request")
            assert engine.ready.wait(2)
            engine.abort_request("request")
            result = future.result(timeout=2)
            terminal = result[0] if mode == "generate" else result
    assert terminal.finished
    assert terminal.outputs[0].finish_reason == "abort"
    assert engine._registry.events == {}
    assert engine._registry.get_record("request") is None
    assert (engine._registry.get_output("request") is None) is (cap == 0)


def test_public_stream_abandonment_cancels_and_releases_output_lease(monkeypatch):
    monkeypatch.setattr(esurge_engine, "get_metrics_collector", lambda: None)
    monkeypatch.setattr(esurge_engine, "log_metrics_summary", lambda: None)
    engine = HostEngine(0)
    stream = engine.stream("hello", request_id="request")
    assert not next(stream).finished
    stream.close()
    assert not engine.scheduler.requests
    assert not engine._registry.events
    assert not engine._registry.outputs
    assert not engine._registry.output_is_retained("request")


@pytest.mark.parametrize("mode", ["generate", "stream", "multimodal"])
def test_public_consumer_start_failure_releases_request_and_output(monkeypatch, mode):
    monkeypatch.setattr(esurge_engine, "get_metrics_collector", lambda: None)
    monkeypatch.setattr(esurge_engine, "log_metrics_summary", lambda: None)
    engine = HostEngine(0)

    def fail(**kwargs):
        raise RuntimeError("startup failed")

    engine._ensure_scheduler_running = fail
    with pytest.raises(RuntimeError, match="startup failed"):
        if mode == "generate":
            engine.generate("hello", request_id="request", use_tqdm=False)
        elif mode == "stream":
            next(engine.stream("hello", request_id="request"))
        else:
            engine.chat([{"role": "user", "content": [{"type": "image", "image": object()}]}], request_id="request")
    assert not engine.scheduler.requests
    assert not engine._registry.events
    assert not engine._registry.outputs
    assert not engine._registry.output_is_retained("request")
