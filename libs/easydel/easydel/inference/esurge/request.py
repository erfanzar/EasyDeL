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

"""Admission data of an eSurge request.

A request's scheduling state (tokens, computed counts, pages, status) lives in surgecore's Rust ``EngineCore``.
Python keeps only what it was admitted with: :class:`AdmissionRecord` carries the prompt ids, the sampling parameters,
the ``n>1`` parent link and the vision-language data, keyed in the engine by the core's request handle. Admission
builds it, the request plane ships it to the owning host, and the plan adapter reads it to build the runner's
``NewRequestData``.

The former ``EngineRequest`` and ``EngineRequestStatus`` are kept as names only: using either raises
:class:`RuntimeError` that points to :class:`AdmissionRecord` and ``EngineCore``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..sampling_params import SamplingParams

if TYPE_CHECKING:
    from .multimodal import MultiModalFeature


@dataclass(eq=False)
class AdmissionRecord:
    """What a request was admitted with; immutable after admission except for the image expansion.

    Attributes:
        request_id: Unique identifier (a child id ``"{parent}-{i}"`` for ``n>1`` samples).
        prompt_token_ids: Prompt token ids (image placeholders expanded once the owner admits it).
        sampling_params: Generation parameters; ``max_tokens`` must be set.
        eos_token_id: Primary EOS id, or ``None``.
        client_index: Client index (always 0 today).
        arrival_time: Arrival time; the priority policy's tie-break. ``None`` stamps the current time.
        priority: Priority (lower first under the priority policy).
        parent_request_id: Parent id of an ``n>1`` sample, ``None`` otherwise.
        sample_index: Sample index of an ``n>1`` child.
        pixel_values: Image pixel values for vision-language models.
        image_grid_thw: Grid shape ``(T, H, W)`` per image.
        pixel_values_videos: Video pixel values.
        video_grid_thw: Grid shape per video.
        mm_features: Multimodal features (their ``mm_hash`` keys the prefix cache).
        images: Raw ``uint8 [H, W, 3]`` images for models that expand image placeholders at admission
            (``prompt_token_ids`` holds one image token per image).
        token_types: ``int64 [len(prompt)]`` image-span token types after expansion, ``None`` otherwise.
        image_spans: Half-open ``(start, stop)`` image spans after expansion.
        stop_reason: The stop string that finished the request, when a parser matched one.
    """

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams | None
    eos_token_id: int | None
    client_index: int = 0
    arrival_time: float | None = None
    priority: int = 0
    parent_request_id: str | None = None
    sample_index: int = 0
    pixel_values: Any | None = None
    image_grid_thw: Any | None = None
    pixel_values_videos: Any | None = None
    video_grid_thw: Any | None = None
    mm_features: list[MultiModalFeature] | None = field(default_factory=list)
    images: list[np.ndarray] | None = None
    token_types: np.ndarray | None = None
    image_spans: list[tuple[int, int]] = field(default_factory=list)
    stop_reason: int | str | None = None

    def __post_init__(self) -> None:
        if self.sampling_params is not None and self.sampling_params.max_tokens is None:
            raise ValueError("sampling_params.max_tokens must not be None")
        if self.arrival_time is None:
            self.arrival_time = time.time()
        if self.mm_features is None:
            self.mm_features = []

    @property
    def num_prompt_tokens(self) -> int:
        """Prompt length."""
        return len(self.prompt_token_ids)

    @property
    def max_tokens(self) -> int:
        """Output-token limit."""
        return int(self.sampling_params.max_tokens) if self.sampling_params is not None else 1

    @property
    def has_vision(self) -> bool:
        """Whether the request carries image or video inputs (pixels or multimodal features)."""
        return self.pixel_values is not None or self.pixel_values_videos is not None or len(self.mm_features) > 0

    def set_expanded_prompt(
        self,
        token_ids: list[int],
        token_types: np.ndarray,
        spans: list[tuple[int, int]],
        features: list[MultiModalFeature],
    ) -> None:
        """Replace the prompt with its image-expanded form (at admission on the owning host).

        Args:
            token_ids: Expanded prompt ids.
            token_types: ``int64 [len(token_ids)]`` image-span token types (``-1`` text).
            spans: Half-open ``(start, stop)`` position range of each image span.
            features: One multimodal feature per image (carries the patches).
        """
        self.prompt_token_ids = list(token_ids)
        self.token_types = token_types
        self.image_spans = list(spans)
        self.mm_features = list(features)
        self.images = None


class _RemovedMeta(type):
    """Metaclass of a removed name: any attribute access raises, as instantiation does."""

    def __getattr__(cls, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(cls.__doc__)


def _removed(name: str) -> type:
    """A placeholder for a request-state name whose state moved into surgecore's ``EngineCore``."""
    message = (
        f"easydel.inference.esurge.request.{name} was removed: request scheduling state lives in surgecore's "
        "Rust EngineCore. Admit requests as easydel.inference.esurge.request.AdmissionRecord."
    )

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(message)

    return _RemovedMeta(name, (), {"__new__": __new__, "__doc__": message})


EngineRequest = _removed("EngineRequest")
EngineRequestStatus = _removed("EngineRequestStatus")

__all__ = ("AdmissionRecord", "EngineRequest", "EngineRequestStatus")
