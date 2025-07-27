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

import time
import typing as tp
from dataclasses import dataclass

from ..sampling_params import SamplingParams

T = tp.TypeVar("T")


class vSurgeMetadata:
    """Tracks timing information for requests processed by vSurge.

    Attributes:
        start_time (float): The Unix timestamp (seconds) when request processing began.
    """

    def __init__(self):
        """Initializes the metadata, capturing the current time as the start time."""
        self.start_time = time.time()


@dataclass
class vSurgeRequest:
    """Represents a request for text completion within the vSurge system.

    Attributes:
        prompt (str): The input prompt for text completion.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): Nucleus sampling probability. Defaults to 0.95.
        top_k (int): Number of highest probability tokens for top-k filtering. Defaults to 0.
        min_p (float): Minimum probability for a token to be considered. Defaults to 0.0.
        n (int): Number of independent samples to generate. Defaults to 1.
        stop (tp.Optional[tp.Union[str, tp.list[str]]]): String or list of strings to
            stop generation if encountered. Defaults to None.
        temperature (float): Sampling temperature. Defaults to 0.7.
        presence_penalty (float): Penalty for token presence. Defaults to 0.0.
        frequency_penalty (float): Penalty for token frequency. Defaults to 0.0.
        repetition_penalty (float): Penalty for repeated tokens. Defaults to 1.0.
        metadata (tp.Optional[vSurgeMetadata]): Metadata associated with the request.
            Auto-initialized if None.
        is_client_side_tokenization (bool): If True, prompt is tokenized and client expects
            token IDs. Defaults to False.
    """

    prompt: str
    sampling_params: SamplingParams
    metadata: vSurgeMetadata | None = None
    is_client_side_tokenization: bool = False

    @classmethod
    def from_sampling_params(cls, prompt: str, sampling_params: SamplingParams) -> vSurgeRequest:
        """Creates a vSurgeRequest from a prompt and SamplingParams.

        Args:
            prompt (str): The input prompt string.
            sampling_params (SamplingParams): An object containing sampling parameters.

        Returns:
            vSurgeRequest: A new vSurgeRequest instance.
        """
        return vSurgeRequest(prompt=prompt, sampling_params=sampling_params)

    def __post_init__(self):
        """Ensures metadata is initialized and validates prompt type."""
        if self.metadata is None:
            self.metadata = vSurgeMetadata()
        assert isinstance(self.prompt, str), "prompt should be a single string"
