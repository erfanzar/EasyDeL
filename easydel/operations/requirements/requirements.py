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

"""Core requirement dataclasses for operation requirements system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .types import CacheType, ExecutionMode, MetadataField

if TYPE_CHECKING:
    pass

__all__ = ["CacheRequirements", "MetadataRequirements", "ModeSpecificRequirements", "OperationRequirements"]


@dataclass(frozen=True)
class MetadataRequirements:
    """Requirements for metadata fields.

    Attributes:
        required: Metadata fields that MUST be present for the operation to work.
        optional: Metadata fields that CAN be used if available but aren't required.
    """

    required: MetadataField = MetadataField.NONE
    optional: MetadataField = MetadataField.NONE

    def __or__(self, other: MetadataRequirements) -> MetadataRequirements:
        """Union of two metadata requirements."""
        return MetadataRequirements(
            required=self.required | other.required,
            optional=self.optional | other.optional,
        )

    def __and__(self, other: MetadataRequirements) -> MetadataRequirements:
        """Intersection of two metadata requirements."""
        return MetadataRequirements(
            required=self.required & other.required,
            optional=self.optional & other.optional,
        )

    @property
    def all_fields(self) -> MetadataField:
        """All fields (required + optional)."""
        return self.required | self.optional

    def is_satisfied_by(self, available: MetadataField) -> bool:
        """Check if all required fields are available.

        Args:
            available: The metadata fields that are available.

        Returns:
            True if all required fields are present in available.
        """
        return (self.required & available) == self.required

    def missing_fields(self, available: MetadataField) -> MetadataField:
        """Get required fields that are missing from available.

        Args:
            available: The metadata fields that are available.

        Returns:
            MetadataField flags for missing required fields.
        """
        return self.required & ~available


@dataclass(frozen=True)
class CacheRequirements:
    """Requirements for cache types.

    Attributes:
        supported: Cache types that the operation can work with.
        preferred: The preferred cache type, if any. Used for optimization hints.
        requires_cache: Whether this operation requires cache at all.
            Set to False for operations that don't need KV cache (e.g., some linear attention).
        cache_view_class: The cache view class this operation uses (e.g., TransformerCacheView).
            Operations return the actual class, which can be used to initialize the cache.
    """

    supported: CacheType = field(default_factory=CacheType.any)
    preferred: CacheType | None = None
    requires_cache: bool = True
    cache_view_class: type | None = None

    def __or__(self, other: CacheRequirements) -> CacheRequirements:
        """Union of two cache requirements (intersection of supported types)."""
        new_supported = self.supported & other.supported
        new_preferred = None
        if self.preferred == other.preferred:
            new_preferred = self.preferred
        elif self.preferred is None:
            new_preferred = other.preferred
        elif other.preferred is None:
            new_preferred = self.preferred
        new_requires_cache = self.requires_cache or other.requires_cache
        new_cache_view_class = None
        if self.cache_view_class == other.cache_view_class:
            new_cache_view_class = self.cache_view_class
        elif self.cache_view_class is None:
            new_cache_view_class = other.cache_view_class
        elif other.cache_view_class is None:
            new_cache_view_class = self.cache_view_class
        return CacheRequirements(
            supported=new_supported,
            preferred=new_preferred,
            requires_cache=new_requires_cache,
            cache_view_class=new_cache_view_class,
        )

    def is_compatible_with(self, cache_type: CacheType) -> bool:
        """Check if a cache type is compatible with these requirements.

        Args:
            cache_type: The cache type to check.

        Returns:
            True if the cache type is supported.
        """
        return self.supported.is_compatible_with(cache_type)


@dataclass(frozen=True)
class OperationRequirements:
    """Complete requirements for an operation.

    Combines metadata and cache requirements with operation identification.

    Attributes:
        metadata: Required and optional metadata fields.
        cache: Supported cache types.
        name: Name of the operation (for debugging/logging).
    """

    metadata: MetadataRequirements = field(default_factory=MetadataRequirements)
    cache: CacheRequirements = field(default_factory=CacheRequirements)
    name: str = ""

    def __or__(self, other: OperationRequirements) -> OperationRequirements:
        """Union of two operation requirements."""
        return OperationRequirements(
            metadata=self.metadata | other.metadata,
            cache=self.cache | other.cache,
            name=f"{self.name}+{other.name}" if self.name and other.name else self.name or other.name,
        )

    @classmethod
    def create(
        cls,
        name: str = "",
        required_metadata: MetadataField = MetadataField.NONE,
        optional_metadata: MetadataField = MetadataField.NONE,
        supported_cache: CacheType | None = None,
        preferred_cache: CacheType | None = None,
        requires_cache: bool = True,
        cache_view_class: type | None = None,
    ) -> OperationRequirements:
        """Convenience factory method for creating requirements.

        Args:
            name: Operation name.
            required_metadata: Required metadata fields.
            optional_metadata: Optional metadata fields.
            supported_cache: Supported cache types (defaults to ANY).
            preferred_cache: Preferred cache type.
            requires_cache: Whether operation requires cache at all.
            cache_view_class: The cache view class this operation uses.

        Returns:
            A new OperationRequirements instance.
        """
        if supported_cache is None:
            supported_cache = CacheType.any()
        return cls(
            metadata=MetadataRequirements(
                required=required_metadata,
                optional=optional_metadata,
            ),
            cache=CacheRequirements(
                supported=supported_cache,
                preferred=preferred_cache,
                requires_cache=requires_cache,
                cache_view_class=cache_view_class,
            ),
            name=name,
        )

    @classmethod
    def default(cls, name: str = "") -> OperationRequirements:
        """Create default requirements (basic metadata, any cache).

        Args:
            name: Operation name.

        Returns:
            Default OperationRequirements with basic metadata and any cache type.
        """
        return cls.create(
            name=name,
            required_metadata=MetadataField.basic(),
            supported_cache=CacheType.any(),
        )

    def with_requires_cache(self, requires_cache: bool) -> OperationRequirements:
        """Create a copy with a modified requires_cache value.

        This is useful for instance-level overrides where an operation
        that normally requires cache should be disabled (e.g., vision encoders).

        Args:
            requires_cache: The new requires_cache value.

        Returns:
            A new OperationRequirements instance with the modified cache requirement.
        """
        new_cache = CacheRequirements(
            supported=self.cache.supported,
            preferred=self.cache.preferred,
            requires_cache=requires_cache,
            cache_view_class=self.cache.cache_view_class if requires_cache else None,
        )
        return OperationRequirements(
            metadata=self.metadata,
            cache=new_cache,
            name=self.name,
        )


@dataclass(frozen=True)
class ModeSpecificRequirements:
    """Requirements that vary by execution mode.

    Some operations have different requirements for prefill vs decode.
    This class allows specifying mode-specific requirements.

    Attributes:
        prefill: Requirements for prefill mode.
        decode: Requirements for decode mode.
        mixed: Requirements for mixed mode (defaults to union of prefill and decode).
    """

    prefill: OperationRequirements = field(default_factory=OperationRequirements.default)
    decode: OperationRequirements = field(default_factory=OperationRequirements.default)
    mixed: OperationRequirements | None = None

    def get(self, mode: ExecutionMode) -> OperationRequirements:
        """Get requirements for a specific execution mode.

        Args:
            mode: The execution mode.

        Returns:
            The requirements for that mode.
        """
        if mode == ExecutionMode.PREFILL:
            return self.prefill
        elif mode == ExecutionMode.DECODE:
            return self.decode
        else:
            # Mixed mode: return union of prefill and decode, or explicit mixed if set
            if self.mixed is not None:
                return self.mixed
            return self.prefill | self.decode

    @classmethod
    def uniform(cls, requirements: OperationRequirements) -> ModeSpecificRequirements:
        """Create mode-specific requirements with same requirements for all modes.

        Args:
            requirements: The requirements to use for all modes.

        Returns:
            ModeSpecificRequirements with uniform requirements.
        """
        return cls(prefill=requirements, decode=requirements, mixed=requirements)
