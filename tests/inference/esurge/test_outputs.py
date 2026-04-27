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

"""Tests for eSurge output structures."""

from easydel.inference.esurge.outputs import (
    LogprobsLists,
    swap_dict_values,
)


class TestSwapDictValues:
    def test_both_keys_exist(self):
        d = {"a": 1, "b": 2}
        swap_dict_values(d, "a", "b")
        assert d == {"a": 2, "b": 1}

    def test_only_key1_exists(self):
        d = {"a": 1}
        swap_dict_values(d, "a", "b")
        assert d == {"b": 1}

    def test_only_key2_exists(self):
        d = {"b": 2}
        swap_dict_values(d, "a", "b")
        assert d == {"a": 2}

    def test_neither_key_exists(self):
        d = {"c": 3}
        swap_dict_values(d, "a", "b")
        assert d == {"c": 3}

    def test_empty_dict(self):
        d = {}
        swap_dict_values(d, "a", "b")
        assert d == {}

    def test_same_key(self):
        d = {"a": 1}
        swap_dict_values(d, "a", "a")
        assert d == {"a": 1}

    def test_none_values(self):
        d = {"a": None, "b": 2}
        swap_dict_values(d, "a", "b")
        assert d == {"a": 2}

    def test_preserves_other_keys(self):
        d = {"a": 1, "b": 2, "c": 3}
        swap_dict_values(d, "a", "b")
        assert d["c"] == 3


class TestLogprobsLists:
    def test_creation(self):
        lp = LogprobsLists(
            logprob_token_ids=[[1, 2], [3, 4]],
            logprobs=[[-0.1, -0.2], [-0.3, -0.4]],
            sampled_token_ranks=[0, 1],
        )
        assert lp.logprob_token_ids == [[1, 2], [3, 4]]
        assert len(lp.logprobs) == 2
        assert lp.sampled_token_ranks == [0, 1]

    def test_slice(self):
        lp = LogprobsLists(
            logprob_token_ids=[[1, 2], [3, 4], [5, 6]],
            logprobs=[[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
            sampled_token_ranks=[0, 1, 2],
        )
        sliced = lp.slice(0, 2)
        assert len(sliced.logprob_token_ids) == 2
        assert sliced.sampled_token_ranks == [0, 1]

    def test_empty(self):
        lp = LogprobsLists(
            logprob_token_ids=[],
            logprobs=[],
            sampled_token_ranks=[],
        )
        assert len(lp.logprob_token_ids) == 0
