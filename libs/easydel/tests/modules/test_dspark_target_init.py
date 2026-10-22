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

"""DSpark embedding / LM-head initialization from a target model."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.modules.dspark.modeling_dspark import DSparkModel


def _param(array):
    return SimpleNamespace(value=array)


def _layer(array):
    return SimpleNamespace(weight=_param(array))


def _drafter(tie_word_embeddings: bool, vocab: int = 5, hidden: int = 3):
    return SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=tie_word_embeddings),
        embed_tokens=_layer(np.zeros((vocab, hidden), dtype=np.float32)),
        lm_head=_layer(np.zeros((hidden, vocab), dtype=np.float32)),
        _copy_parameter_value=DSparkModel._copy_parameter_value,
    )


def _target(vocab: int = 5, hidden: int = 3):
    embed = np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden)
    head = (np.arange(hidden * vocab, dtype=np.float32) + 100).reshape(hidden, vocab)
    return SimpleNamespace(get_input_embeddings=lambda: _layer(embed), get_lm_head=lambda: _layer(head)), embed, head


@pytest.mark.cpu_ok
def test_initialize_copies_embeddings_and_untied_head():
    drafter = _drafter(tie_word_embeddings=False)
    target, embed, head = _target()
    DSparkModel.initialize_embeddings_and_head_from_target(drafter, target)
    np.testing.assert_array_equal(drafter.embed_tokens.weight.value, embed)
    np.testing.assert_array_equal(drafter.lm_head.weight.value, head)


@pytest.mark.cpu_ok
def test_initialize_skips_head_when_tied():
    drafter = _drafter(tie_word_embeddings=True)
    target, embed, _ = _target()
    DSparkModel.initialize_embeddings_and_head_from_target(drafter, target)
    np.testing.assert_array_equal(drafter.embed_tokens.weight.value, embed)
    assert not np.any(drafter.lm_head.weight.value)


@pytest.mark.cpu_ok
def test_initialize_rejects_shape_mismatch():
    drafter = _drafter(tie_word_embeddings=False, vocab=7)
    target, _, _ = _target()
    with pytest.raises(ValueError, match="shape mismatch"):
        DSparkModel.initialize_embeddings_and_head_from_target(drafter, target)
