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

"""eSurge installs the tokenizer's engram compressed token map for engram models (no array compute).

The released DeepSeek-V4.1 checkpoint hashes n-grams over a 99,092-entry compressed vocabulary while its vocab is
129,280, so the runner's identity-map fallback would raise on the first request. The engine must build the map from
its tokenizer, reject a tokenizer that does not match the config, and leave other models alone.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_ok


class _Runner:
    def __init__(self, model):
        self.model = model
        self.maps = []

    def set_side_input_tokenizer_state(self, token_map):
        self.maps.append(np.asarray(token_map))


def _engine(*, needs_engram=True, compressed=3, vocab=5, tokenizer="fast"):
    from easydel.layers.engram import EngramModelMixin

    config = SimpleNamespace(engram_layer_ids=[1] if needs_engram else [], engram_compressed_vocab_size=compressed)
    config.vocab_size = vocab

    class EngramModel(EngramModelMixin):
        # specs stubbed: this test has no module tree, only the mixin's tokenizer hook is under test
        def esurge_side_input_specs(self):
            return {"engram_hash_ids": ((1, 4), np.int32)} if self.config.engram_layer_ids else {}

    model = EngramModel()
    model.config = config
    tok = SimpleNamespace(backend_tokenizer=object()) if tokenizer == "fast" else None
    return SimpleNamespace(runner=_Runner(model), tokenizer=tok)


def _install(engine):
    from easydel.inference.esurge.esurge_engine import eSurge

    eSurge._install_side_input_tokenizer_state(engine)


@pytest.fixture
def fake_map(monkeypatch):
    import easydel.layers.engram._hashing as engram_hash

    token_map = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)  # compressed vocab 3
    monkeypatch.setattr(engram_hash, "build_engram_token_map", lambda tokenizer: token_map)
    return token_map


def test_installs_tokenizer_map_for_engram_models(fake_map):
    engine = _engine()
    _install(engine)
    assert len(engine.runner.maps) == 1
    np.testing.assert_array_equal(engine.runner.maps[0], fake_map)


def test_rejects_tokenizer_that_disagrees_with_config(fake_map):
    with pytest.raises(ValueError, match="compressed vocab size 3"):
        _install(_engine(compressed=7))


def test_models_without_engram_are_untouched(fake_map):
    engine = _engine(needs_engram=False)
    _install(engine)
    assert engine.runner.maps == []


def test_no_tokenizer_falls_back_only_when_identity_map_fits(fake_map):
    engine = _engine(compressed=5, vocab=5, tokenizer=None)
    _install(engine)  # identity map is valid: the runner builds it lazily
    assert engine.runner.maps == []
    with pytest.raises(ValueError, match="no fast tokenizer"):
        _install(_engine(compressed=3, vocab=5, tokenizer=None))
