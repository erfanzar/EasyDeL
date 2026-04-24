from __future__ import annotations

import jax.numpy as jnp
import pytest

from easydel.inference.esurge.esurge_engine import eSurge


class _AbortModelLoad(RuntimeError):
    pass


class _DummyTokenizer:
    name_or_path = "dummy-tokenizer"
    pad_token_id = 0
    eos_token_id = 1


class _DummyWorkerManager:
    def __init__(self, *_args, **_kwargs):
        self.tokenizer_endpoint = "ipc://tokenizer"
        self.detokenizer_endpoint = "ipc://detokenizer"
        self._startup_timeout = 1.0

    def start(self, **_kwargs):
        return object(), object()


def _patch_engine_bootstrap(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    monkeypatch.setattr(
        "easydel.inference.esurge.esurge_engine.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "easydel.inference.esurge.esurge_engine.WorkerManager",
        _DummyWorkerManager,
    )

    def _fake_from_pretrained(*_args, **kwargs):
        captured["config_kwargs"] = kwargs["config_kwargs"]
        raise _AbortModelLoad()

    monkeypatch.setattr(
        "easydel.modules.auto.AutoEasyDeLModelForCausalLM.from_pretrained",
        _fake_from_pretrained,
    )


def test_esurge_allows_config_kwargs_kvdtype_override(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    _patch_engine_bootstrap(monkeypatch, captured)

    with pytest.raises(_AbortModelLoad):
        eSurge(
            model="dummy-model",
            tokenizer="dummy-tokenizer",
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            hbm_utilization=0.5,
            page_size=16,
            config_kwargs={"kvdtype": jnp.float16},
        )

    assert captured["config_kwargs"]["kvdtype"] == jnp.float16


def test_esurge_top_level_kvdtype_override_wins(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    _patch_engine_bootstrap(monkeypatch, captured)

    with pytest.raises(_AbortModelLoad):
        eSurge(
            model="dummy-model",
            tokenizer="dummy-tokenizer",
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            hbm_utilization=0.5,
            page_size=16,
            kvdtype=jnp.float8_e4m3fn,
            config_kwargs={"kvdtype": jnp.float16},
        )

    assert captured["config_kwargs"]["kvdtype"] == jnp.float8_e4m3fn