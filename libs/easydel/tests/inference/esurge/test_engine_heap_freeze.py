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

"""eSurge's startup-heap freeze (``freeze_startup_heap``): held while the serving loop runs, released on terminate.

The hold-count test is pure Python. The engine tests build a tiny engine (computation tests: the suite's conftest
refuses them on a CPU backend).
"""

from __future__ import annotations

import gc
import weakref

import jax
import pytest
import spectrax as spx
from easydel.inference.esurge import esurge_engine
from jax import numpy as jnp

try:
    from tests.inference.esurge.runtime_pass._tiny_runtime_common import load_test_tokenizer
except ImportError:  # pragma: no cover - direct invocation
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "runtime_pass"))
    from _tiny_runtime_common import load_test_tokenizer  # pyright: ignore[reportImplicitRelativeImport]

MAX_MODEL_LEN = 128


class _Node:
    def __init__(self) -> None:
        self.other: _Node | None = None


@pytest.mark.cpu_ok
def test_frozen_garbage_is_collected_only_after_the_last_release():
    assert esurge_engine._HEAP_FREEZE["holders"] == 0
    a, b = _Node(), _Node()
    a.other, b.other = b, a
    ref = weakref.ref(a)
    esurge_engine._freeze_heap()
    esurge_engine._freeze_heap()
    del a, b
    gc.collect()
    assert ref() is not None, "a frozen cycle was collected"
    esurge_engine._release_heap_freeze()
    gc.collect()
    assert ref() is not None, "the first of two holders unfroze the heap"
    esurge_engine._release_heap_freeze()
    gc.collect()
    assert ref() is None, "the last release did not return the frozen cycle to the collector"
    assert gc.get_freeze_count() == 0
    esurge_engine._release_heap_freeze()  # a release without a hold is a no-op
    assert esurge_engine._HEAP_FREEZE["holders"] == 0


@pytest.fixture(scope="module")
def tiny_llama():
    import easydel as ed

    tokenizer = load_test_tokenizer()
    config = ed.LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=MAX_MODEL_LEN,
        kvdtype=jnp.float32,
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sharding_axis_dims=(1, 1, 1, 1, jax.device_count(), 1),
    )
    with config.mesh:
        model = ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    return model, tokenizer


def _engine(model, tokenizer, **runtime):
    return esurge_engine.eSurge(
        model=model,
        processor=tokenizer,
        runtime={"max_model_len": MAX_MODEL_LEN, "max_num_seqs": 2, **runtime},
        cache={"page_size": 16, "hbm_utilization": 0.05, "max_cache_tokens": 2048, "enable_prefix_caching": False},
        parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
    )


def test_serving_engine_holds_the_freeze_while_its_loop_runs(tiny_llama):
    model, tokenizer = tiny_llama
    assert gc.get_freeze_count() == 0
    engine = _engine(model, tokenizer)
    try:
        assert engine.runtime_config.freeze_startup_heap is True
        assert gc.get_freeze_count() > 0, "the started engine did not freeze the startup heap"
        engine.pause()
        assert gc.get_freeze_count() == 0, "a paused engine kept the heap frozen"
        engine.resume()
        assert gc.get_freeze_count() > 0, "the resumed engine did not freeze the heap again"
    finally:
        engine.terminate()
    assert gc.get_freeze_count() == 0, "terminate did not release the freeze"
    assert esurge_engine._HEAP_FREEZE["holders"] == 0


def test_engine_with_the_freeze_off_and_get_esurge_leave_gc_alone(tiny_llama):
    model, tokenizer = tiny_llama
    engine = _engine(model, tokenizer, freeze_startup_heap=False)
    try:
        assert gc.get_freeze_count() == 0
    finally:
        engine.terminate()
    # get_esurge builds the engines trainers generate through: they share the process with training.
    shared = model.get_esurge(
        tokenizer=tokenizer,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=2,
        hbm_utilization=0.05,
        page_size=16,
        max_cache_tokens=2048,
        enable_prefix_caching=False,
        silent_mode=True,
    )
    try:
        assert shared.runtime_config.freeze_startup_heap is False
        assert gc.get_freeze_count() == 0
    finally:
        shared.terminate()
