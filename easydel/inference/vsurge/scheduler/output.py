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

import typing
from dataclasses import dataclass

import jax
import numpy as np
import numpy.typing as npt

from ...sampling_params import SamplingParams
from ..request_type import Request


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    mm_inputs: list[dict[str, jax.Array]]
    mm_hashes: list[str]
    mm_positions: list[typing.Any]
    sampling_params: SamplingParams
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> NewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_inputs=request.mm_inputs,
            mm_hashes=request.mm_hashes,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )

    def __repr__(self):
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"mm_inputs={self.mm_inputs},"
            f"mm_hashes={self.mm_hashes},"
            f"mm_positions={self.mm_positions},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            ")"
        )


@dataclass
class CachedRequestData:
    req_id: str
    resumed_from_preemption: bool
    new_token_ids: list[int]
    new_block_ids: tuple[list[int], ...]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        resumed_from_preemption: bool,
        new_token_ids: list[int],
        new_block_ids: tuple[list[int], ...],
    ) -> CachedRequestData:
        return cls(
            req_id=request.request_id,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: list[CachedRequestData]
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]
    free_encoder_input_ids: list[tuple[str, int]]
    structured_output_request_ids: dict[str, int]
    grammar_bitmask: npt.NDArray[np.int32] | None
