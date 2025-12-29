# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Type definitions for operation requirements system."""

from enum import Enum, Flag, auto

__all__ = [
    "CacheType",
    "ExecutionMode",
    "MetadataField",
]


class ExecutionMode(Enum):
    """Execution mode for inference operations.

    Operations can declare different requirements based on whether they're
    running in prefill, decode, or mixed mode.
    """

    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


class MetadataField(Flag):
    """Metadata fields that operations can require.

    Operations declare which metadata fields they need, and the inference
    engine builds only the required fields.

    Flags can be combined with | operator:
        required = MetadataField.SEQ_LENS | MetadataField.POSITIONS

    Membership can be checked with `in`:
        if MetadataField.SEQ_LENS in required: ...
    """

    NONE = 0

    # Core sequence information
    SEQ_LENS = auto()
    """Sequence lengths for each request in batch."""

    CONTEXT_LENS = auto()
    """Context lengths (total KV cache length) for each request."""

    POSITIONS = auto()
    """Position IDs for each token."""

    # Ragged/paged attention fields
    QUERY_START_LOC = auto()
    """Starting locations for queries in ragged batch format."""

    PAGES_TABLES = auto()
    """Page/block tables mapping logical to physical pages."""

    SLOT_MAPPING = auto()
    """Slot mapping for RPA v2 style paged attention."""

    REQUEST_DISTRIBUTION = auto()
    """Request distribution for RPA v3 style attention."""

    # Recurrent/state space model fields
    HAS_INITIAL_STATE = auto()
    """Boolean indicating if initial state is provided."""

    STATE_INDICES = auto()
    """Indices mapping requests to their state slots."""

    # Output selection
    LOGITS_INDICES = auto()
    """Indices for selecting which positions to compute logits for."""

    @classmethod
    def basic(cls) -> "MetadataField":
        """Basic metadata for simple attention operations."""
        return cls.SEQ_LENS | cls.POSITIONS | cls.LOGITS_INDICES

    @classmethod
    def ragged(cls) -> "MetadataField":
        """Metadata for ragged batch format."""
        return cls.basic() | cls.QUERY_START_LOC | cls.CONTEXT_LENS

    @classmethod
    def paged_v2(cls) -> "MetadataField":
        """Metadata for RPA v2 paged attention."""
        return cls.ragged() | cls.PAGES_TABLES | cls.SLOT_MAPPING

    @classmethod
    def paged_v3(cls) -> "MetadataField":
        """Metadata for RPA v3 paged attention."""
        return cls.ragged() | cls.PAGES_TABLES | cls.REQUEST_DISTRIBUTION

    @classmethod
    def recurrent(cls) -> "MetadataField":
        """Metadata for recurrent/state space models."""
        return cls.basic() | cls.HAS_INITIAL_STATE | cls.STATE_INDICES


class CacheType(Flag):
    """Cache types that operations can support.

    Operations declare which cache types they're compatible with.
    The inference engine validates cache compatibility at initialization.
    """

    NONE = 0

    TRANSFORMER = auto()
    """Standard transformer KV cache (TransformerCacheView)."""

    RAGGED_PAGES = auto()
    """Ragged paged cache for continuous batching (RaggedPagesCacheView)."""

    RECURRENT = auto()
    """Recurrent state cache for SSMs (RecurrentCacheView)."""

    HYBRID = auto()
    """Hybrid cache combining multiple cache types (HybridCacheView)."""

    @classmethod
    def any(cls) -> "CacheType":
        """Any cache type - operation is cache-agnostic."""
        return cls.TRANSFORMER | cls.RAGGED_PAGES | cls.RECURRENT | cls.HYBRID

    @classmethod
    def attention(cls) -> "CacheType":
        """Cache types suitable for attention operations."""
        return cls.TRANSFORMER | cls.RAGGED_PAGES | cls.HYBRID

    def is_compatible_with(self, other: "CacheType") -> bool:
        """Check if this cache type is compatible with another.

        Args:
            other: The cache type to check compatibility with.

        Returns:
            True if there's any overlap between the cache types.
        """
        return bool(self & other)
