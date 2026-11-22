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

"""vWhisper config, decoder-id, chunking, audio-io, core, and server regression tests."""

from __future__ import annotations

import asyncio
import copy
import inspect
import io
import os
import sys
import tempfile
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import fastapi.dependencies.utils as fastapi_dep_utils
import httpx
import numpy as np
import pytest
import requests
from easydel.inference import vwhisper
from easydel.inference.logits_process import ForceTokensLogitsProcessor
from easydel.inference.vwhisper import generation, server, utils
from easydel.inference.vwhisper.config import vWhisperInferenceConfig
from easydel.inference.vwhisper.core import vWhisperInference
from easydel.inference.vwhisper.generation import get_decoder_input_ids
from easydel.inference.vwhisper.server import ResponseFormat, WhisperModel, create_whisper_app
from easydel.inference.vwhisper.utils import chunk_iter_with_batch, process_audio_input
from fastapi import HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

pytestmark = pytest.mark.cpu_ok


def _generation_config(is_multilingual: bool) -> SimpleNamespace:
    return SimpleNamespace(
        is_multilingual=is_multilingual,
        lang_to_id={"<|en|>": 11},
        task_to_id={"transcribe": 22, "translate": 23},
        no_timestamps_token_id=99,
    )


def _dummy_feature_extractor(sampling_rate: int = 16000, chunk_length: int = 30):
    class _Extractor:
        def __init__(self):
            self.sampling_rate = sampling_rate
            self.chunk_length = chunk_length
            self.calls = 0

        def __call__(self, audio, sampling_rate=None, return_tensors="np"):
            self.calls += 1
            if isinstance(audio, list):
                batch = len(audio)
                features = np.zeros((batch, 80, 4), dtype=np.float32)
            else:
                features = np.zeros((1, 80, 4), dtype=np.float32)
            return {"input_features": features}

    return _Extractor()


def test_config_accepts_documented_constructor_arguments():
    config = vWhisperInferenceConfig(
        batch_size=8, max_length=448, return_timestamps=True, language="en", task="translate"
    )
    assert (config.batch_size, config.max_length) == (8, 448)
    assert config.return_timestamps is True
    assert config.language == "en"
    assert config.task == "translate"
    assert vWhisperInferenceConfig().language is None


def test_config_hash_depends_on_language():
    assert hash(vWhisperInferenceConfig(language="en")) == hash(vWhisperInferenceConfig(language="en"))
    assert hash(vWhisperInferenceConfig(language="en")) != hash(vWhisperInferenceConfig(language="fr"))


def test_config_hash_and_eq_normalize_return_timestamps_and_custom_logits_processor():
    class _CustomProcessor:
        def __init__(self, scale: float, weights: np.ndarray | None = None):
            self.scale = scale
            self.weights = weights

    c_none = vWhisperInferenceConfig(return_timestamps=None)
    c_false = vWhisperInferenceConfig(return_timestamps=False)
    c_true = vWhisperInferenceConfig(return_timestamps=True)
    assert c_none == c_false
    assert hash(c_none) == hash(c_false)
    assert c_none.cache_key() == c_false.cache_key()
    assert c_false != c_true
    assert hash(c_false) != hash(c_true)

    p1 = _CustomProcessor(0.5, np.array([1.0, 2.0], dtype=np.float32))
    p2 = _CustomProcessor(0.5, np.array([1.0, 2.0], dtype=np.float32))
    p3 = _CustomProcessor(1.5, np.array([1.0, 2.0], dtype=np.float32))
    assert repr(p1) != repr(p2)  # Different 0x... object addresses.

    cfg1 = vWhisperInferenceConfig(logits_processor=[p1])
    cfg2 = vWhisperInferenceConfig(logits_processor=[p2])
    cfg3 = vWhisperInferenceConfig(logits_processor=[p3])
    assert "0x" not in repr(cfg1.cache_key())
    assert cfg1 == cfg2
    assert hash(cfg1) == hash(cfg2)
    assert cfg1 != cfg3
    assert hash(cfg1) != hash(cfg3)


def test_decoder_ids_force_language_task_and_no_timestamps():
    ids = get_decoder_input_ids(
        model_config=SimpleNamespace(),
        generation_config=_generation_config(True),
        language="en",
        task="transcribe",
        return_timestamps=False,
    )
    assert ids == [(1, 11), (2, 22), (3, 99)]


def test_decoder_ids_is_multilingual_override_wins_over_generation_config():
    ids = get_decoder_input_ids(
        model_config=SimpleNamespace(),
        generation_config=_generation_config(False),
        language="en",
        task="translate",
        return_timestamps=True,
        is_multilingual=True,
    )
    assert ids == [(1, 11), (2, 23)]


def test_decoder_ids_do_not_duplicate_existing_no_timestamps_token():
    config = _generation_config(True)
    config.task_to_id["transcribe"] = config.no_timestamps_token_id
    ids = get_decoder_input_ids(
        model_config=SimpleNamespace(), generation_config=config, language="en", task="transcribe"
    )
    assert ids == [(1, 11), (2, 99)]


def test_decoder_ids_non_multilingual_return_timestamps_false_and_true():
    ids_no_ts = get_decoder_input_ids(
        model_config=SimpleNamespace(),
        generation_config=_generation_config(False),
        return_timestamps=False,
    )
    assert ids_no_ts == [(1, 99)]

    ids_with_ts = get_decoder_input_ids(
        model_config=SimpleNamespace(),
        generation_config=_generation_config(False),
        return_timestamps=True,
    )
    assert ids_with_ts == []


def test_non_multilingual_generate_with_timestamps_works_through_force_generate(monkeypatch):
    from easydel.modules._base import BaseConditionalGenerationModule
    from easydel.modules.whisper.modeling_whisper import WhisperForConditionalGeneration
    from jax import numpy as jnp
    from transformers import GenerationConfig

    def fake_base_generate(self, inputs, generation_config, logits_processor=None, **kwargs):
        return SimpleNamespace(sequences=jnp.zeros((1, 4), dtype=jnp.int32), logits_processor=logits_processor)

    monkeypatch.setattr(BaseConditionalGenerationModule, "generate", fake_base_generate)

    model = object.__new__(WhisperForConditionalGeneration)
    gen_cfg = GenerationConfig(
        eos_token_id=2,
        is_multilingual=False,
        no_timestamps_token_id=50362,
        begin_suppress_tokens=[220, 50256],
        max_length=16,
    )
    object.__setattr__(model, "generation_config", gen_cfg)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(max_source_positions=1500, is_multilingual=False, vocab_size=51864, mesh=nullcontext()),
    )

    monkeypatch.setattr(generation.spx, "export", lambda m: ("graph", "state"))
    monkeypatch.setattr(generation.spx, "bind", lambda g, s: model)

    engine = vWhisperInference(
        model=model,
        tokenizer=SimpleNamespace(),
        processor=SimpleNamespace(feature_extractor=_dummy_feature_extractor()),
    )

    captured_decoder_ids: list[object] = []

    def unwrapped_generate(**kwargs):
        captured_decoder_ids.append(kwargs["decoder_input_ids"])
        return inspect.unwrap(generation._compiled_generate)(**kwargs)

    engine.generate_function = unwrapped_generate

    # Case 1: is_multilingual=False, return_timestamps=False -> forces {1: 50362}
    out_no_ts = engine._generate(np.zeros((1, 80, 4), dtype=np.float32), return_timestamps=False)
    assert out_no_ts.shape == (1, 4)
    assert captured_decoder_ids[-1] == {1: 50362}

    # Case 2: is_multilingual=False, return_timestamps=True -> avoids empty {} in ForceTokensLogitsProcessor
    out_with_ts = engine._generate(np.zeros((1, 80, 4), dtype=np.float32), return_timestamps=True)
    assert out_with_ts.shape == (1, 4)
    assert captured_decoder_ids[-1] == {0: None}

    # Verify _compiled_generate itself also normalizes empty {} / [] / None safely.
    for empty_ids in ({}, [], None):
        raw_out = inspect.unwrap(generation._compiled_generate)(
            engine.graphdef,
            engine.graphstate,
            engine.inference_config,
            jnp.zeros((1, 80, 4), dtype=jnp.float32),
            empty_ids,
            True,
        )
        force_procs = [p for p in raw_out.logits_processor if isinstance(p, ForceTokensLogitsProcessor)]
        assert len(force_procs) == 1
        scores = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        for cur_len in (0, 1, 2):
            np.testing.assert_allclose(
                np.asarray(force_procs[0](jnp.zeros((1, 1), dtype=jnp.int32), scores, cur_len)),
                np.asarray(scores),
            )


def test_chunk_length_equal_to_total_stride_is_rejected_with_value_error():
    fake_self = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    audio = np.zeros(32000, dtype=np.float32)
    with pytest.raises(ValueError, match="Chunk length"):
        next(vWhisperInference._process_model_inputs(fake_self, audio, chunk_length_s=1.0, stride_length_s=0.5))


def test_chunk_iter_with_batch_empty_audio_returns_immediately():
    extractor = _dummy_feature_extractor()
    batches = list(
        chunk_iter_with_batch(
            audio_array=np.zeros(0, dtype=np.float32),
            chunk_length=480000,
            stride_left=80000,
            stride_right=80000,
            batch_size=4,
            feature_extractor=extractor,
        )
    )
    assert batches == []
    assert extractor.calls == 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"chunk_length": 100, "stride_left": 50, "stride_right": 50, "batch_size": 1}, "step"),
        ({"chunk_length": 100, "stride_left": 60, "stride_right": 50, "batch_size": 1}, "step"),
        ({"chunk_length": 100, "stride_left": 10, "stride_right": 10, "batch_size": 0}, "batch_size"),
        ({"chunk_length": 100, "stride_left": 10, "stride_right": 10, "batch_size": -2}, "batch_size"),
        ({"chunk_length": 100, "stride_left": 10, "stride_right": 10, "batch_size": None}, "batch_size"),
        ({"chunk_length": 100, "stride_left": 10, "stride_right": 10, "batch_size": True}, "batch_size"),
        ({"chunk_length": 0, "stride_left": 0, "stride_right": 0, "batch_size": 1}, "chunk_length"),
        ({"chunk_length": 100, "stride_left": -1, "stride_right": 0, "batch_size": 1}, "non-negative"),
    ],
)
def test_chunk_iter_with_batch_validates_arguments(kwargs, match):
    extractor = _dummy_feature_extractor()
    with pytest.raises(ValueError, match=match):
        list(
            chunk_iter_with_batch(
                audio_array=np.zeros(1000, dtype=np.float32),
                feature_extractor=extractor,
                **kwargs,
            )
        )


@pytest.mark.parametrize(
    "num_samples,expected_strides",
    [
        (160_000, [(160_000, 0, 0)]),  # 10s
        (320_000, [(320_000, 0, 0)]),  # 20s (exact step)
        (360_000, [(360_000, 0, 0)]),  # 22.5s (regression: previously emitted invalid tail chunk)
        (480_000, [(480_000, 0, 0)]),  # 30s (exact chunk_length; right stride must be 0)
        (480_001, [(480_000, 0, 80_000), (160_001, 80_000, 0)]),  # 30s + 1 sample
        (720_000, [(480_000, 0, 80_000), (400_000, 80_000, 0)]),  # 45s (regression for (40s, 50s))
        (800_000, [(480_000, 0, 80_000), (480_000, 80_000, 0)]),  # 50s
        (960_000, [(480_000, 0, 80_000), (480_000, 80_000, 80_000), (320_000, 80_000, 0)]),  # 60s
    ],
)
def test_chunk_iter_with_batch_boundary_durations(num_samples, expected_strides):
    extractor = _dummy_feature_extractor()
    audio = np.zeros(num_samples, dtype=np.float32)
    batches = list(
        chunk_iter_with_batch(
            audio_array=audio,
            chunk_length=480_000,
            stride_left=80_000,
            stride_right=80_000,
            batch_size=2,
            feature_extractor=extractor,
        )
    )
    strides = [stride for batch in batches for stride in batch["stride"]]
    assert strides == expected_strides
    # Universal invariants: stride_left < chunk_len and usable spans partition [0, num_samples].
    for chunk_len, sl, sr in strides:
        assert chunk_len > sl
        assert sl + sr <= chunk_len
    assert sum(chunk_len - sl - sr for chunk_len, sl, sr in strides) == num_samples


def test_process_audio_input_scales_stride_before_validation_on_resampling(monkeypatch):
    def fake_resample(y, *, orig_sr, target_sr):
        target_len = round(len(y) * target_sr / orig_sr)
        return np.zeros(target_len, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(resample=fake_resample))
    extractor = _dummy_feature_extractor(sampling_rate=16000)

    # 1s at 48kHz with (0.25s, 0.25s) stride -> resamples to 16000 with stride (4000, 4000).
    # Unscaled stride sum (24000) > resampled len (16000), but scaled sum (8000) <= 16000.
    audio, stride = process_audio_input(
        {"array": np.zeros(48000, dtype=np.float32), "sampling_rate": 48000, "stride": (12000, 12000)},
        extractor,
    )
    assert audio.shape == (16000,)
    assert stride == (16000, 4000, 4000)

    # Scaled stride sum (10000 + 10000 = 20000) > 16000 must still be rejected.
    with pytest.raises(ValueError, match="Stride is too large"):
        process_audio_input(
            {"array": np.zeros(48000, dtype=np.float32), "sampling_rate": 48000, "stride": (30000, 30000)},
            extractor,
        )


def test_process_audio_input_http_passes_timeout_and_calls_raise_for_status(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeResponse:
        def __init__(self, status_ok: bool):
            self.content = b"fake-audio-bytes"
            self._status_ok = status_ok
            self.raised_checked = False

        def raise_for_status(self):
            self.raised_checked = True
            if not self._status_ok:
                raise requests.HTTPError("404 Not Found")

    ok_resp = _FakeResponse(status_ok=True)

    def fake_get(url, *, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return ok_resp

    monkeypatch.setattr(utils.requests, "get", fake_get)
    monkeypatch.setattr(
        "transformers.pipelines.audio_utils.ffmpeg_read",
        lambda payload, sr: np.zeros(sr, dtype=np.float32),
    )

    extractor = _dummy_feature_extractor(sampling_rate=16000)
    audio, stride = process_audio_input("https://example.com/clip.wav", extractor)
    assert audio.shape == (16000,)
    assert stride is None
    assert captured == {"url": "https://example.com/clip.wav", "timeout": 30.0}
    assert ok_resp.raised_checked is True

    # HTTP error response must raise instead of feeding error body to ffmpeg.
    monkeypatch.setattr(utils.requests, "get", lambda url, *, timeout=None: _FakeResponse(status_ok=False))
    with pytest.raises(requests.HTTPError, match="404"):
        process_audio_input("https://example.com/missing.wav", extractor, timeout=5.0)


def test_non_chunked_stride_wrapped_and_postprocessed_without_type_error(monkeypatch):
    from jax import numpy as jnp

    captured_outputs: list[dict] = []

    class _FakeTokenizer:
        def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
            captured_outputs.extend(copy.deepcopy(model_outputs))
            return "hello", {}

    model_gen_cfg = _generation_config(False)
    model_gen_cfg.max_length = 32
    model = SimpleNamespace(
        config=SimpleNamespace(max_source_positions=1500, is_multilingual=False),
        generation_config=model_gen_cfg,
        mesh=nullcontext(),
    )
    monkeypatch.setattr("easydel.inference.vwhisper.core.spx.export", lambda m: ("g", "s"))

    caller_cfg = vWhisperInferenceConfig(batch_size=None, return_timestamps=None)
    engine = vWhisperInference(
        model=model,
        tokenizer=_FakeTokenizer(),
        processor=SimpleNamespace(feature_extractor=_dummy_feature_extractor(sampling_rate=16000, chunk_length=30)),
        inference_config=caller_cfg,
    )

    # Caller config must NOT be mutated in place.
    assert caller_cfg.generation_config is None
    assert engine.inference_config is not caller_cfg
    assert engine.inference_config.generation_config is model_gen_cfg

    seen_generate_calls: list[dict] = []

    def fake_generate_fn(**kwargs):
        seen_generate_calls.append(kwargs)
        batch = kwargs["input_features"].shape[0]
        return SimpleNamespace(sequences=jnp.ones((batch, 5), dtype=jnp.int32))

    engine.generate_function = fake_generate_fn

    # Non-chunked path with explicit stride + batch_size=None + return_timestamps=None
    result = engine.generate(
        {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000, "stride": (1600, 3200)},
        chunk_length_s=0,
    )
    assert result == {"text": "hello"}
    assert len(captured_outputs) == 1
    assert captured_outputs[0]["stride"] == (1.0, 0.1, 0.2)
    assert type(captured_outputs[0]["tokens"]) is np.ndarray
    assert captured_outputs[0]["tokens"].shape == (1, 5)
    assert seen_generate_calls[0]["return_timestamps"] is False

    # Chunked path with batch_size=None also succeeds and yields host np.ndarray tokens.
    captured_outputs.clear()
    result2 = engine.generate(np.zeros(360_000, dtype=np.float32), chunk_length_s=30.0)
    assert result2 == {"text": "hello"}
    assert len(captured_outputs) == 1
    assert type(captured_outputs[0]["tokens"]) is np.ndarray


def test_single_batch_process_converts_jax_array_to_host_numpy():
    from jax import numpy as jnp

    fake_engine = SimpleNamespace(
        inference_config=vWhisperInferenceConfig(batch_size=None),
        _generate=lambda **kwargs: jnp.arange(12, dtype=jnp.int32).reshape(3, 4),
    )
    out = vWhisperInference._single_batch_process(
        fake_engine,
        model_inputs={"input_features": np.zeros((2, 80, 4), dtype=np.float32), "stride": [(1, 0, 0), (1, 0, 0)]},
        batch_size=3,
    )
    assert type(out["tokens"]) is np.ndarray
    assert out["tokens"].shape == (2, 1, 4)
    np.testing.assert_array_equal(out["tokens"][:, 0, :], np.arange(8, dtype=np.int32).reshape(2, 4))


def test_whisper_model_singleton_no_poisoning_mismatch_and_thread_safety(monkeypatch):
    WhisperModel.reset()
    try:
        attempts = 0

        def flaky_init(self):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient download failure")
            time.sleep(0.02)
            self.inference = lambda **kwargs: {"text": "ok"}

        monkeypatch.setattr(WhisperModel, "_initialize", flaky_init)

        # First init fails -> singleton must NOT be poisoned.
        with pytest.raises(RuntimeError, match="transient download failure"):
            WhisperModel("openai/whisper-tiny", dtype="float32")
        assert WhisperModel._instance is None
        assert WhisperModel() is None

        # Concurrent initialization succeeds once and shares one instance.
        results: list[WhisperModel] = []
        barrier = threading.Barrier(6)

        def worker():
            barrier.wait()
            results.append(WhisperModel("openai/whisper-tiny", dtype="float32"))

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert attempts == 2  # 1 failed attempt + 1 successful init across all 6 threads.
        assert len({id(r) for r in results}) == 1
        instance = results[0]

        # No-arg lookup returns existing singleton regardless of default dtype.
        assert WhisperModel() is instance

        # Model or dtype mismatch raises ValueError instead of silently returning wrong model.
        with pytest.raises(ValueError, match="already initialized"):
            WhisperModel("openai/whisper-base", dtype="float32")
        with pytest.raises(ValueError, match="already initialized"):
            WhisperModel("openai/whisper-tiny", dtype="bfloat16")
    finally:
        WhisperModel.reset()


def _get_route(app, path: str) -> APIRoute:
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            return route
    raise AssertionError(f"Route not found: {path}")


def test_server_formats_granularities_cors_filename_none_and_upload_limit(monkeypatch):
    WhisperModel.reset()
    monkeypatch.setattr(fastapi_dep_utils, "ensure_multipart_is_installed", lambda: None)
    recorded_calls: list[dict] = []

    def fake_init(self):
        def fake_inference(**kwargs):
            recorded_calls.append(kwargs)
            assert os.path.exists(kwargs["audio_input"])
            out: dict[str, object] = {"text": "Hello world. How are you?"}
            if kwargs.get("return_timestamps"):
                out["chunks"] = [
                    {"timestamp": (0.0, 1.5), "text": "Hello world."},
                    {"timestamp": (1.5, 3.25), "text": " How are you?"},
                ]
            return out

        self.inference = fake_inference

    monkeypatch.setattr(WhisperModel, "_initialize", fake_init)
    try:
        app = create_whisper_app(model_name="openai/whisper-tiny", dtype="float32", max_upload_bytes=64)
        client = TestClient(app)

        # CORS must NOT pair allow_origins=["*"] with allow_credentials=True.
        cors_resp = client.get("/", headers={"Origin": "https://example.com"})
        assert cors_resp.status_code == 200
        assert cors_resp.headers.get("access-control-allow-origin") == "*"
        assert cors_resp.headers.get("access-control-allow-credentials") != "true"

        transcribe_route = _get_route(app, "/v1/audio/transcriptions")
        translate_route = _get_route(app, "/v1/audio/translations")
        assert transcribe_route.response_model is None
        assert translate_route.response_model is None

        transcribe_ep = transcribe_route.endpoint
        translate_ep = translate_route.endpoint

        async def exercise():
            # response_format="text" returns PlainTextResponse (text/plain), not JSON.
            resp_text = await transcribe_ep(
                file=UploadFile(filename="clip.wav", file=io.BytesIO(b"1234")),
                model="openai/whisper-tiny",
                prompt=None,
                response_format=ResponseFormat.text,
                temperature=0.0,
                language=None,
                timestamp_granularities=None,
                timestamp_granularities_bracket=None,
            )
            assert isinstance(resp_text, PlainTextResponse)
            assert resp_text.media_type == "text/plain"
            assert resp_text.body.decode() == "Hello world. How are you?"
            assert recorded_calls[-1]["return_timestamps"] is False

            # filename=None does not raise TypeError in either endpoint.
            resp_no_name = await transcribe_ep(
                file=UploadFile(filename=None, file=io.BytesIO(b"1234")),
                model="openai/whisper-tiny",
                prompt=None,
                response_format=ResponseFormat.json,
                temperature=0.0,
                language=None,
                timestamp_granularities=None,
                timestamp_granularities_bracket=None,
            )
            assert resp_no_name == {"text": "Hello world. How are you?"}

            # response_format="srt" returns formatted SRT text/plain with timestamps enabled.
            resp_srt = await transcribe_ep(
                file=UploadFile(filename="clip.wav", file=io.BytesIO(b"1234")),
                model="openai/whisper-tiny",
                prompt=None,
                response_format=ResponseFormat.srt,
                temperature=0.0,
                language=None,
                timestamp_granularities=None,
                timestamp_granularities_bracket=None,
            )
            assert isinstance(resp_srt, PlainTextResponse)
            srt_text = resp_srt.body.decode()
            assert "1\n00:00:00,000 --> 00:00:01,500\nHello world." in srt_text
            assert "2\n00:00:01,500 --> 00:00:03,250\nHow are you?" in srt_text
            assert recorded_calls[-1]["return_timestamps"] is True

            # response_format="vtt" returns WebVTT text/plain on translations endpoint too.
            resp_vtt = await translate_ep(
                file=UploadFile(filename=None, file=io.BytesIO(b"1234")),
                model="openai/whisper-tiny",
                prompt=None,
                response_format=ResponseFormat.vtt,
                temperature=0.0,
                timestamp_granularities=None,
                timestamp_granularities_bracket=None,
            )
            assert isinstance(resp_vtt, PlainTextResponse)
            vtt_text = resp_vtt.body.decode()
            assert vtt_text.startswith("WEBVTT\n\n")
            assert "00:00:00.000 --> 00:00:01.500\nHello world." in vtt_text
            assert "00:00:01.500 --> 00:00:03.250\nHow are you?" in vtt_text
            assert recorded_calls[-1]["task"] == "translate"
            assert recorded_calls[-1]["return_timestamps"] is True

            # response_format="verbose_json" includes segments, timestamps, duration, language.
            payload = await transcribe_ep(
                file=UploadFile(filename="clip.wav", file=io.BytesIO(b"1234")),
                model="openai/whisper-tiny",
                prompt=None,
                response_format=ResponseFormat.verbose_json,
                temperature=0.0,
                language="en",
                timestamp_granularities=None,
                timestamp_granularities_bracket=None,
            )
            assert payload["task"] == "transcribe"
            assert payload["language"] == "en"
            assert payload["duration"] == 3.25
            assert payload["text"] == "Hello world. How are you?"
            assert len(payload["segments"]) == 2
            assert payload["segments"][0]["start"] == 0.0
            assert payload["segments"][0]["end"] == 1.5
            assert payload["segments"][1]["timestamp"] == [1.5, 3.25]

            # timestamp_granularities=["segment"] enables return_timestamps and emits segments in json.
            resp_seg = await transcribe_ep(
                file=UploadFile(filename="clip.wav", file=io.BytesIO(b"1234")),
                model="openai/whisper-tiny",
                prompt=None,
                response_format=ResponseFormat.json,
                temperature=0.0,
                language=None,
                timestamp_granularities=["segment"],
                timestamp_granularities_bracket=None,
            )
            assert recorded_calls[-1]["return_timestamps"] is True
            assert len(resp_seg["segments"]) == 2

            # timestamp_granularities[]=["segment"] (OpenAI SDK style) also works.
            resp_seg_bracket = await translate_ep(
                file=UploadFile(filename="clip.wav", file=io.BytesIO(b"1234")),
                model="openai/whisper-tiny",
                prompt=None,
                response_format=ResponseFormat.json,
                temperature=0.0,
                timestamp_granularities=None,
                timestamp_granularities_bracket=["segment"],
            )
            assert recorded_calls[-1]["return_timestamps"] is True
            assert len(resp_seg_bracket["segments"]) == 2

            # Oversized upload returns HTTP 413 (not 500) and cleans up temp file.
            created_paths: list[str] = []
            orig_ntf = tempfile.NamedTemporaryFile

            def tracking_ntf(*args, **kwargs):
                handle = orig_ntf(*args, **kwargs)
                created_paths.append(handle.name)
                return handle

            monkeypatch.setattr(server.tempfile, "NamedTemporaryFile", tracking_ntf)
            with pytest.raises(HTTPException) as exc_info:
                await transcribe_ep(
                    file=UploadFile(filename="big.wav", file=io.BytesIO(b"x" * 128)),
                    model="openai/whisper-tiny",
                    prompt=None,
                    response_format=ResponseFormat.json,
                    temperature=0.0,
                    language=None,
                    timestamp_granularities=None,
                    timestamp_granularities_bracket=None,
                )
            assert exc_info.value.status_code == 413
            assert len(created_paths) == 1
            assert not os.path.exists(created_paths[0])

        asyncio.run(exercise())

        # Verify all temp files created in earlier requests were cleaned up.
        for call in recorded_calls:
            assert not os.path.exists(call["audio_input"])
    finally:
        WhisperModel.reset()


def test_server_uploaded_temp_file_handles_none_filename_and_cancelled_error(monkeypatch):
    created_paths: list[str] = []
    orig_ntf = tempfile.NamedTemporaryFile

    def tracking_ntf(*args, **kwargs):
        handle = orig_ntf(*args, **kwargs)
        created_paths.append(handle.name)
        return handle

    monkeypatch.setattr(server.tempfile, "NamedTemporaryFile", tracking_ntf)

    # 1) filename=None must not crash os.path.splitext and must clean up afterward.
    async def run_none_filename():
        upload = UploadFile(filename=None, file=io.BytesIO(b"hello-audio"))
        async with server._uploaded_temp_file(upload, max_upload_bytes=1024) as path:
            assert os.path.exists(path)
            with open(path, "rb") as f:
                assert f.read() == b"hello-audio"
        return path

    temp_path = asyncio.run(run_none_filename())
    assert not os.path.exists(temp_path)

    # 2) CancelledError during file.read() must still delete the temp file immediately.
    class _CancellingUploadFile:
        filename = None

        def __init__(self):
            self.calls = 0

        async def read(self, size: int = -1) -> bytes:
            self.calls += 1
            if self.calls == 1:
                return b"partial"
            raise asyncio.CancelledError

    created_paths.clear()

    async def run_cancelled():
        async with server._uploaded_temp_file(_CancellingUploadFile(), max_upload_bytes=1024):
            pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_cancelled())
    assert len(created_paths) == 1
    assert not os.path.exists(created_paths[0])

    # 3) Env var configures max upload bytes.
    monkeypatch.setenv(server.MAX_UPLOAD_BYTES_ENV_VAR, "2048")
    assert server._resolve_max_upload_bytes(None) == 2048

    # 4) Oversized check directly via _uploaded_temp_file raises HTTP 413 and unlinks.
    created_paths.clear()

    async def run_oversized():
        upload = UploadFile(filename=None, file=io.BytesIO(b"a" * 32))
        async with server._uploaded_temp_file(upload, max_upload_bytes=16):
            pass

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(run_oversized())
    assert exc_info.value.status_code == 413
    assert len(created_paths) == 1
    assert not os.path.exists(created_paths[0])


def test_server_offloads_blocking_inference_off_event_loop(monkeypatch):
    WhisperModel.reset()
    monkeypatch.setattr(fastapi_dep_utils, "ensure_multipart_is_installed", lambda: None)
    entered = threading.Event()
    release = threading.Event()
    inference_thread_id: list[int] = []

    def fake_init(self):
        def blocking_inference(**kwargs):
            inference_thread_id.append(threading.get_ident())
            entered.set()
            assert release.wait(timeout=5.0)
            return {"text": "offloaded"}

        self.inference = blocking_inference

    monkeypatch.setattr(WhisperModel, "_initialize", fake_init)

    async def exercise():
        main_thread_id = threading.get_ident()
        app = create_whisper_app(model_name="openai/whisper-tiny", dtype="float32")
        transcribe_ep = _get_route(app, "/v1/audio/transcriptions").endpoint
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            post_task = asyncio.create_task(
                transcribe_ep(
                    file=UploadFile(filename="clip.wav", file=io.BytesIO(b"1234")),
                    model="openai/whisper-tiny",
                    prompt=None,
                    response_format=ResponseFormat.text,
                    temperature=0.0,
                    language=None,
                    timestamp_granularities=None,
                    timestamp_granularities_bracket=None,
                )
            )
            # Wait until blocking inference is running in worker thread.
            while not entered.is_set():
                await asyncio.sleep(0.005)

            # Event loop remains responsive while inference is blocked.
            health = await client.get("/")
            assert health.status_code == 200
            assert health.json()["model"] == "openai/whisper-tiny"

            release.set()
            resp = await post_task
            assert isinstance(resp, PlainTextResponse)
            assert resp.body.decode() == "offloaded"
            assert inference_thread_id == [inference_thread_id[0]]
            assert inference_thread_id[0] != main_thread_id

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        WhisperModel.reset()


def test_vwhisper_init_docstring_matches_public_api_and_cli_flags():
    doc = vwhisper.__doc__ or ""
    assert "--model-id" not in doc
    assert "--model openai/whisper-base" in doc
    assert "engine.transcribe(" not in doc
    assert "engine.generate(" in doc
    assert hasattr(vWhisperInference, "generate")
    assert not hasattr(vWhisperInference, "transcribe")
