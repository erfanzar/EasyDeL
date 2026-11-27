"""Array-free regressions for scheduler lifetime, request outputs and streaming stops."""

import os
import shutil
import tempfile
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
from easydel.inference.sampling_params import SamplingParams
from easydel.modules.auto import AutoEasyDeLModelForCausalLM
from easydel.modules.auto.auto_modeling import PreTrainedLoading
from surgecore import CoreClosed

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
        raise CoreClosed("closed")

    def info(*args):
        if failure == "setup":
            raise ValueError("setup failed")

    loop = EngineLoop(
        core=SimpleNamespace(schedule=schedule, closed_error=CoreClosed, num_running=0, num_waiting=0),
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


class _HostCore:
    """Array-free engine-core boundary: live admission records by id."""

    num_running = 0
    num_waiting = 0

    def __init__(self):
        self.requests = {}

    def record_of(self, request_id):
        return self.requests.get(request_id)

    def is_live(self, request_id):
        return request_id in self.requests

    def children_of(self, parent_request_id):
        return [rid for rid, r in self.requests.items() if r.parent_request_id == parent_request_id]

    def finish_requests(self, ids, reason):
        return [rid for rid in ids if self.requests.pop(rid, None) is not None]


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
        self.core = _HostCore()
        self._admission = admission(self._registry, self.submit_requests)
        self._multimodal_manager = NoCacheMediaManager()
        self.ready = threading.Event()

    def submit_requests(self, requests):
        self.core.requests.update({request.request_id: request for request in requests})

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
    assert not engine.core.requests
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
    assert not engine.core.requests
    assert not engine._registry.events
    assert not engine._registry.outputs
    assert not engine._registry.output_is_retained("request")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_esurge_terminate_reaps_real_tokenizer_worker_while_pause_and_weight_update_preserve_it(tmp_path):
    """Real subprocess regression for P1 worker leak on eSurge.terminate() + pause/resume/initiate/weight-update."""
    from easydel.workers.esurge.pipeline.worker_manager import WorkerManager
    from tokenizers import Tokenizer as HfTokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tok_dir = tmp_path / "tok"
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=HfTokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    hf_tok.save_pretrained(tok_dir)

    ipc_dir = tempfile.mkdtemp(prefix="ed_term_", dir="/tmp")
    manager = WorkerManager(
        str(tok_dir),
        tokenizer_kwargs={"local_files_only": True},
        startup_timeout=60.0,
        ipc_dir=ipc_dir,
    )
    try:
        tok_client, detok_client = manager.start(detokenizer_max_states=16, tokenizer_endpoint=None)
        first_proc = manager._tokenizer_process
        first_endpoint = manager.tokenizer_endpoint
        assert first_proc is not None and first_proc.poll() is None
        assert first_endpoint is not None and first_endpoint.startswith("ipc://")
        first_pid = first_proc.pid
        first_sock = first_endpoint[len("ipc://") :]

        engine = eSurge.__new__(eSurge)
        engine._info = lambda *args, **kwargs: None
        engine._scheduler_lock = threading.RLock()
        engine._scheduler_running = True
        engine._engine_loop = None
        engine._worker_replay_thread = None
        engine._worker_stop_event = threading.Event()
        engine._monitoring = None
        engine._profiling_active = False
        engine._step_coordinator = None
        engine._request_plane = None
        engine._paused = False
        engine._kv_cache_valid = True
        engine._holds_heap_freeze = False
        engine._configured_tokenizer_endpoint = None
        engine._worker_manager = manager
        engine._tokenizer_client = tok_client
        engine._detokenizer_client = detok_client
        engine._tokenizer_endpoint = first_endpoint
        engine._admission = SimpleNamespace(_tokenizer_client=tok_client)
        engine._output_pipeline = SimpleNamespace(
            _detokenizer_client=detok_client, start=lambda: None, stop=lambda: None
        )
        engine.worker_config = SimpleNamespace(detokenizer_max_states=16)
        engine.cache_config = SimpleNamespace(destroy_pages_on_pause=False)
        engine.runtime_config = SimpleNamespace(
            freeze_startup_heap=False,
            overlap_execution=False,
            async_scheduling=False,
            runner_verbose=False,
        )
        engine.runner = SimpleNamespace(
            shutdown=lambda: None,
            executor_manager=SimpleNamespace(kv_pages=object()),
        )
        engine.core = SimpleNamespace(
            num_running=0,
            num_waiting=0,
            closed_error=CoreClosed,
            schedule=lambda: SimpleNamespace(total_num_scheduled_tokens=0, finished_req_ids=()),
        )
        engine._registry = RequestRegistry()
        engine._stop_idle_monitor = lambda: None
        engine._start_idle_monitor = lambda: None
        engine._touch_activity = lambda: None
        engine._reset_runner_state_if_idle = lambda reason: None
        engine._install_signal_diagnostics = lambda: None
        engine._enqueue_engine_outputs = lambda outputs: None
        engine._drain_parser_stop_requests_locked = lambda: None
        engine._abort_scheduler_due_to_error = lambda exc: None
        engine._update_scheduler_heartbeat = lambda: None
        engine._handle_profiling_step = lambda: None
        engine._is_nonrecoverable_scheduler_error = lambda exc: False

        # 1. pause() must keep the tokenizer worker alive and usable across pause/resume.
        engine.pause()
        assert engine._paused is True
        assert manager.is_running is True
        assert first_proc.poll() is None
        assert _pid_alive(first_pid)
        assert engine._tokenizer_client.tokenize("r-paused", "hello") == [1]

        # 2. terminate() after pause() (early_return path, scheduler already stopped) must shut down and reap it
        # without holding _scheduler_lock.
        lock_free_during_shutdown: list[bool] = []
        orig_shutdown = manager.shutdown

        def _check_lock_free_shutdown():
            acquired_from_other_thread: list[bool] = []

            def _probe():
                acquired = engine._scheduler_lock.acquire(blocking=False)
                if acquired:
                    engine._scheduler_lock.release()
                acquired_from_other_thread.append(acquired)

            probe = threading.Thread(target=_probe)
            probe.start()
            probe.join(timeout=2.0)
            lock_free_during_shutdown.extend(acquired_from_other_thread)
            orig_shutdown()

        manager.shutdown = _check_lock_free_shutdown
        try:
            engine.terminate()
        finally:
            manager.shutdown = orig_shutdown

        assert lock_free_during_shutdown == [True]
        assert manager.is_running is False
        assert engine._tokenizer_endpoint is None
        assert first_proc.poll() is not None
        assert not _pid_alive(first_pid)
        with pytest.raises(ChildProcessError):
            os.waitpid(first_pid, os.WNOHANG)
        assert not os.path.exists(first_sock)

        # 3. Calling public initiate() after terminate() transparently restarts the tokenizer worker,
        # starts a fresh EngineLoop, and rebinds clients on engine, admission, and output_pipeline.
        engine.initiate()
        assert engine.alive is True
        second_proc = manager._tokenizer_process
        second_endpoint = manager.tokenizer_endpoint
        assert second_proc is not None and second_proc.poll() is None
        assert second_proc.pid != first_pid
        assert engine._tokenizer_client is engine._admission._tokenizer_client
        assert engine._detokenizer_client is engine._output_pipeline._detokenizer_client
        assert engine._tokenizer_client.tokenize("r-restarted", "world") == [2]

        # 4. Normal running terminate() path stops the live loop and shuts down + reaps the restarted worker.
        engine.terminate()
        assert engine.alive is False
        assert manager.is_running is False
        assert second_proc.poll() is not None
        assert not _pid_alive(second_proc.pid)
        with pytest.raises(ChildProcessError):
            os.waitpid(second_proc.pid, os.WNOHANG)
        assert second_endpoint is not None and not os.path.exists(second_endpoint[len("ipc://") :])

        # 5. Calling initiate() on a closed engine is rejected without respawning workers.
        engine._closed = True
        with pytest.raises(RuntimeError, match="closed eSurge engine"):
            engine.initiate()
        assert manager.is_running is False
    finally:
        manager.shutdown()
        shutil.rmtree(ipc_dir, ignore_errors=True)


def test_esurge_update_model_weights_preserves_pipeline_workers_and_monitoring(monkeypatch):
    """update_model_weights() stops the scheduler without killing tokenizer workers or active monitoring."""
    engine = eSurge.__new__(eSurge)
    engine._info = lambda *args, **kwargs: None
    engine._closed = False
    engine._weight_version = 0
    engine._scheduler_lock = threading.RLock()
    engine._scheduler_running = True
    engine._engine_loop = None
    engine._worker_replay_thread = None
    engine._step_coordinator = None
    engine._request_plane = None
    engine._multimodal_manager = None
    engine._kv_cache_valid = True
    engine._scheduler_max_num_batched_tokens = 32
    engine._registry = RequestRegistry()
    engine._request_lock = engine._registry.request_lock
    engine._output_lock = engine._registry.output_lock
    engine._active_requests = engine._registry.records
    engine._request_outputs = engine._registry.outputs
    engine._finished_request_ids = engine._registry.finished_ids
    engine._request_events = engine._registry.events
    engine._stop_idle_monitor = lambda: None
    engine._drop_heap_freeze = lambda: None
    engine._stop_engine_output_worker = lambda: None
    engine._reset_runner_state_if_idle = lambda reason: None
    engine._log_cache_event = lambda event, extra=None: None

    worker_events: list[str] = []
    monitoring_events: list[str] = []
    engine._worker_manager = SimpleNamespace(
        drain_workers=lambda: worker_events.append("drain"),
        shutdown=lambda: worker_events.append("shutdown"),
    )
    engine._monitoring = SimpleNamespace(monitoring_active=True, has_active_resources=True)
    engine.stop_monitoring = lambda: monitoring_events.append("stop")
    engine.core = SimpleNamespace(num_running=0, num_waiting=0, shutdown=lambda: None)
    engine._build_core = lambda: SimpleNamespace(num_running=0, num_waiting=0, shutdown=lambda: None)
    engine.runner = SimpleNamespace(
        shutdown=lambda: None,
        update_model_weights=lambda **kwargs: None,
        executor_manager=SimpleNamespace(
            has_compiled_variants=lambda: True,
            kv_pages=object(),
            graphdef="gd",
            graphother="go",
        ),
    )
    initiated: list[bool] = []
    engine.initiate = lambda: initiated.append(True)
    monkeypatch.setattr(engine, "_resolve_weight_update_components", lambda *args: ("gd", "gs", "go"))

    version = engine.update_model_weights(graphstate="gs", restart_scheduler=True)

    assert version == 1
    assert worker_events == ["drain"]
    assert monitoring_events == []
    assert initiated == [True]
