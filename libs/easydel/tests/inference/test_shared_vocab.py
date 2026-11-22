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

"""``get_vocab()`` must be paid once per tokenizer, not once per request.

Reasoning and tool parsers are constructed per admitted request and read the
vocabulary in ``__init__``. A ``cached_property`` on the parser therefore caches
nothing across requests, and ``get_vocab()`` materializes the whole vocabulary
as a dict -- ~32 ms at 129k tokens. Profiling DeepSeek-V4 serving measured
64 ms per admitted request, 2.06 s of a 3.91 s run, none of it on the model's
critical path.
"""

import pytest
from easydel.inference.reasoning.abstract_reasoning import shared_vocab


class FakeTokenizer:
    """Counts how often the expensive vocabulary build is requested."""

    def __init__(self, size: int = 64):
        self._vocab = {f"tok{i}": i for i in range(size)}
        self._vocab["<think>"] = size
        self._vocab["</think>"] = size + 1
        self.calls = 0

    def get_vocab(self):
        self.calls += 1
        return dict(self._vocab)


class SlottedTokenizer(FakeTokenizer):
    """A tokenizer that refuses attribute assignment."""

    __slots__ = ()

    def __setattr__(self, name, value):
        if name in {"_vocab", "calls"}:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(name)


def test_vocab_is_built_once_per_tokenizer():
    tok = FakeTokenizer()
    first = shared_vocab(tok)
    for _ in range(10):
        assert shared_vocab(tok) is first
    assert tok.calls == 1, f"get_vocab() ran {tok.calls} times; must be once"


def test_separate_tokenizers_do_not_share():
    """Two models in one process must not see each other's vocabulary."""
    a, b = FakeTokenizer(size=8), FakeTokenizer(size=9)
    va, vb = shared_vocab(a), shared_vocab(b)
    assert va is not vb
    assert len(va) != len(vb)


def test_tokenizer_that_forbids_attributes_still_works():
    """Caching is an optimization; a tokenizer that rejects it must not break."""
    tok = SlottedTokenizer()
    first = shared_vocab(tok)
    second = shared_vocab(tok)
    assert first == second
    assert tok.calls == 2, "uncacheable tokenizer falls back to recomputing"


def test_parsers_constructed_per_request_do_not_rebuild_the_vocab():
    """The actual regression: N requests must not cost N vocabulary builds."""
    basic = pytest.importorskip("easydel.inference.reasoning.basic_parsers")
    parser_cls = next(
        (
            obj
            for obj in vars(basic).values()
            if isinstance(obj, type) and hasattr(obj, "start_token") and hasattr(obj, "end_token")
        ),
        None,
    )
    if parser_cls is None:
        pytest.skip("no basic reasoning parser exposed")

    tok = FakeTokenizer()
    for _ in range(16):
        parser_cls(tok)
    assert tok.calls == 1, f"16 requests rebuilt the vocab {tok.calls} times"
