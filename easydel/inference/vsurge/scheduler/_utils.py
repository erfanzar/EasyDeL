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

import dataclasses
import enum
import typing

from jax import numpy as jnp

from easydel.layers.caching import PagesMetadata

if typing.TYPE_CHECKING:
    from ..utils import ActiveRequest
    from .page_manager import PageManager


@dataclasses.dataclass
class PageAllocation:
    """Tracks page allocations for a request"""

    page_indices: list[int]
    num_pages: int
    sequence_id: int


@dataclasses.dataclass
class PagedScheduleResult:
    """Result of a scheduling decision"""

    prefill_infos: list[PrefillScheduleInfo]
    decode_info: DecodeScheduleInfo
    updated_page_manager: PageManager
    should_decode: bool
    should_prefill: bool


@dataclasses.dataclass
class ScheduleDecision:
    """Result of a simple scheduling decision"""

    prefill_requests: list[tuple[int, ActiveRequest]]  # (slot, request) pairs
    decode_slots: list[int]  # slots ready for decode
    should_decode: bool
    should_prefill: bool


@dataclasses.dataclass
class PrefillScheduleInfo:
    """Information about a scheduled prefill operation"""

    slot: int
    request: ActiveRequest
    seq_id: int
    token_ids: jnp.ndarray
    page_batch_info: PagesMetadata | None


@dataclasses.dataclass
class DecodeScheduleInfo:
    """Information about scheduled decode operations"""

    active_slots: list[int]
    active_requests: dict[int, ActiveRequest]
    sequence_ids: jnp.ndarray
    page_batch_info: PagesMetadata | None


@dataclasses.dataclass
class ScheduleResult:
    """Complete scheduling decision"""

    prefill_info: list[PrefillScheduleInfo]
    decode_info: DecodeScheduleInfo | None
    updated_page_manager: PageManager | None
    should_prefill: bool
    should_decode: bool


@enum.unique
class SchedulePolicy(enum.Enum):
    """Scheduling policies for request batching"""

    OFFLINE = enum.auto()
    ONLINE = enum.auto()
    ADAPTIVE = enum.auto()


class RequestPriority(enum.Enum):
    """Priority levels for request scheduling"""

    HIGH = 0
    NORMAL = 1
    LOW = 2
