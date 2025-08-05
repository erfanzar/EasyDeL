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

import jax
from eformer.pytree import auto_pytree

from ...sampling_params import SamplingParams


@auto_pytree
class CachedRequestState:
    """Represents the state of a single request, compatible with JAX."""

    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    generator: jax.random.PRNGKey
    page_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]
    num_prompt_tokens: int = -1

    def __post_init__(self):
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        else:
            return self.output_token_ids[idx - self.num_prompt_tokens]
