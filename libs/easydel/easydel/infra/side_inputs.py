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

"""Neutral contracts for model-declared host-built per-token side inputs.

Models and reusable layers declare these contracts independently of serving orchestration.
eSurge's providers and scheduling helpers live in ``easydel.inference.esurge.multimodal.side_inputs``.
"""

from __future__ import annotations

import typing as tp

import numpy as np


class SideInputSpec(tp.NamedTuple):
    """Static description of one per-token side input.

    Attributes:
        per_token_shape: Trailing shape of one token's row (the full host array is ``[T, *per_token_shape]``).
        dtype: NumPy dtype of the array.
        fill: Value of padded rows.
    """

    per_token_shape: tuple[int, ...]
    dtype: tp.Any
    fill: int | float = 0


class SideInputWindow(tp.NamedTuple):
    """The scheduled token window of one request in a step.

    Attributes:
        token_ids: The request's token history row (prompt then generated tokens; positions ``< end`` are real).
        token_types: The request's prompt token types (``None`` for text-only requests).
        start: First scheduled position.
        end: One past the last scheduled position.
        num_tokens: The request's known length (prompt plus accepted outputs, speculative drafts excluded); ``-1``
            when unknown. Feeds the ``tokens_to_end`` side input.
    """

    token_ids: np.ndarray
    token_types: np.ndarray | None
    start: int
    end: int
    num_tokens: int = -1


class SideInputProvider(tp.Protocol):
    """Host builder of a model's per-token side inputs."""

    def build(self, windows: tp.Sequence[SideInputWindow], num_tokens: int) -> dict[str, np.ndarray]:
        """Arrays ``[num_tokens, *per_token_shape]`` per spec name; window rows are packed back to back in order."""
        ...


__all__ = ["SideInputProvider", "SideInputSpec", "SideInputWindow"]
