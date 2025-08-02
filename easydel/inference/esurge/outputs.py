# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import NamedTuple, TypeVar

from jax import Array
from jax import numpy as jnp

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


class LogprobsLists(NamedTuple):
    logprob_token_ids: list[list[int]]
    logprobs: list[list[float]]
    sampled_token_ranks: list[int]

    def slice(self, start: int, end: int):
        return LogprobsLists(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.sampled_token_ranks[start:end],
        )


class LogprobsTensors(NamedTuple):
    logprob_token_ids: Array

    logprobs: Array

    selected_token_ranks: Array

    def tolists(self):
        return LogprobsLists(
            self.logprob_token_ids.tolist(),
            self.logprobs.tolist(),
            self.selected_token_ranks.tolist(),
        )

    @staticmethod
    def empty(num_positions: int, num_tokens_per_position: int) -> LogprobsTensors:
        logprob_token_ids = jnp.empty((num_positions, num_tokens_per_position), dtype=jnp.int32)
        logprobs = jnp.empty_like(logprob_token_ids, dtype=jnp.float32)
        selected_token_ranks = jnp.empty(num_positions, dtype=jnp.int32)
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]
    spec_token_ids: list[list[int]] | None
    logprobs: LogprobsLists | None
    prompt_logprobs_dict: dict[str, LogprobsTensors | None]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    num_nans_in_logits: dict[str, int] | None = None


def swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K) -> None:
    """
    Helper function to swap values for two keys
    """
    v1 = obj.get(key1)
    v2 = obj.get(key2)
    if v1 is not None:
        obj[key2] = v1
    else:
        obj.pop(key2, None)
    if v2 is not None:
        obj[key1] = v2
    else:
        obj.pop(key1, None)
