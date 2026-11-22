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

"""Array-free public projection and postprocessor lifetime contracts."""

import gc
import weakref
from types import SimpleNamespace

import pytest
from easydel.modules._base._features import LogitCapFeature
from easydel.modules.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from easydel.modules.gemma3.modeling_gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from easydel.modules.gemma4 import modeling_gemma4
from easydel.modules.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration
from easydel.modules.gidd.modeling_gidd import GiddForDiffusionLM

pytestmark = pytest.mark.cpu_ok

HEADS = [
    Gemma2ForCausalLM,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    GiddForDiffusionLM,
]


def _head(head_class, *, tied=False, soft_cap=None, clip=None):
    # Allocate the real task type without constructing a backbone or any arrays.
    head = head_class.__new__(head_class)
    text = SimpleNamespace(final_logit_softcapping=soft_cap, tie_word_embeddings=tied)
    config = SimpleNamespace(
        final_logit_softcapping=soft_cap,
        tie_word_embeddings=tied,
        text_config=text,
        get_text_config=lambda: text,
    )
    object.__setattr__(head, "config", config)
    object.__setattr__(head, "precision", None)
    object.__setattr__(head, "_logit_cap_feature", LogitCapFeature(clip) if clip is not None else None)
    return head


@pytest.mark.parametrize("head_class", HEADS, ids=lambda cls: cls.__name__)
def test_disabled_postprocessing_is_explicit_identity(head_class):
    head = _head(head_class)
    assert head.get_lm_logit_postprocess() is None


@pytest.mark.parametrize("head_class", [Gemma4ForCausalLM, GiddForDiffusionLM], ids=lambda cls: cls.__name__)
def test_projection_only_heads_do_not_infer_unrelated_base_clipping(head_class):
    head = _head(head_class, clip=0.125)
    head.config.logit_scale = 99.0
    assert head.get_lm_logit_postprocess() is None


@pytest.mark.parametrize("head_class", HEADS[:-1], ids=lambda cls: cls.__name__)
def test_postprocessor_does_not_keep_model_or_parameter_tree_alive(head_class):
    head = _head(head_class, soft_cap=2.0, clip=1.0)
    ref = weakref.ref(head)
    postprocess = head.get_lm_logit_postprocess()
    del head
    gc.collect()
    assert callable(postprocess)
    assert ref() is None


@pytest.mark.parametrize("head_class", HEADS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_identity_projection_interfaces_preserve_tied_weight_selection(head_class, tied, monkeypatch):
    head = _head(head_class, tied=tied)
    hidden, own_logits, tied_logits, weight_override = object(), object(), object(), object()

    def dot(states, weight, *, precision=None):
        assert states is hidden
        return tied_logits if weight is weight_override else own_logits

    # Replace the numerical boundary with host sentinels; the test checks public
    # head outputs without allocating arrays, including Gemma4's native tied dot.
    monkeypatch.setattr(modeling_gemma4.jnp, "dot", dot)

    class Projection:
        def native_forward(self, states, *, w=None):
            assert states is hidden
            return tied_logits if w is weight_override else own_logits

        __call__ = native_forward

    class Embedding:
        weight = SimpleNamespace(value=SimpleNamespace(T=weight_override))
        dtype = None

        @staticmethod
        def promote_dtype(values, *, dtype):
            return values

        def attend(self, states):
            assert states is hidden
            return tied_logits

    projection, embedding = Projection(), Embedding()
    object.__setattr__(head, "lm_head", projection)
    object.__setattr__(head, "get_lm_head", lambda: projection)
    object.__setattr__(head, "get_embedding", lambda: embedding)
    expected = tied_logits if tied else own_logits
    project = head.make_lm_head_fn(vocab_shard_stage=0)
    assert project(hidden) is expected
    if head_class is GiddForDiffusionLM:
        assert head.compute_lm_logits(hidden) is expected
    else:
        assert head.apply_lm_head(hidden) is expected
