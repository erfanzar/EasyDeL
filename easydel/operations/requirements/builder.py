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

"""Fluent builder API for operation requirements."""

from __future__ import annotations

from .requirements import (
    CacheRequirements,
    MetadataRequirements,
    ModeSpecificRequirements,
    OperationRequirements,
)
from .types import CacheType, ExecutionMode, MetadataField

__all__ = ["ModeSpecificBuilder", "RequirementsBuilder"]


class RequirementsBuilder:
    """Fluent builder for creating OperationRequirements.

    Example usage:
        >>> reqs = (RequirementsBuilder("ragged_page_attention_v3")
        ...     .require_metadata(MetadataField.QUERY_START_LOC | MetadataField.PAGES_TABLES)
        ...     .require_metadata(MetadataField.REQUEST_DISTRIBUTION)
        ...     .optional_metadata(MetadataField.LOGITS_INDICES)
        ...     .support_cache(CacheType.RAGGED_PAGES)
        ...     .prefer_cache(CacheType.RAGGED_PAGES)
        ...     .build())
    """

    def __init__(self, name: str = ""):
        """Initialize the builder.

        Args:
            name: Name of the operation.
        """
        self._name = name
        self._required_metadata = MetadataField.NONE
        self._optional_metadata = MetadataField.NONE
        self._supported_cache = CacheType.any()
        self._preferred_cache: CacheType | None = None
        self._requires_cache: bool = True
        self._cache_view_class: type | None = None

    def require_metadata(self, fields: MetadataField) -> RequirementsBuilder:
        """Add required metadata fields.

        Args:
            fields: Metadata fields that are required.

        Returns:
            Self for method chaining.
        """
        self._required_metadata |= fields
        return self

    def optional_metadata(self, fields: MetadataField) -> RequirementsBuilder:
        """Add optional metadata fields.

        Args:
            fields: Metadata fields that are optional.

        Returns:
            Self for method chaining.
        """
        self._optional_metadata |= fields
        return self

    def support_cache(self, cache_types: CacheType) -> RequirementsBuilder:
        """Set supported cache types.

        Note: This replaces any previously set cache types.
        Use `add_cache_support()` to add to existing.

        Args:
            cache_types: Cache types that are supported.

        Returns:
            Self for method chaining.
        """
        self._supported_cache = cache_types
        return self

    def add_cache_support(self, cache_types: CacheType) -> RequirementsBuilder:
        """Add additional supported cache types.

        Args:
            cache_types: Cache types to add to supported set.

        Returns:
            Self for method chaining.
        """
        self._supported_cache |= cache_types
        return self

    def prefer_cache(self, cache_type: CacheType) -> RequirementsBuilder:
        """Set preferred cache type.

        Args:
            cache_type: The preferred cache type.

        Returns:
            Self for method chaining.
        """
        self._preferred_cache = cache_type
        return self

    def requires_cache(self, value: bool = True) -> RequirementsBuilder:
        """Set whether this operation requires cache.

        Args:
            value: True if operation requires cache, False otherwise.

        Returns:
            Self for method chaining.
        """
        self._requires_cache = value
        return self

    def no_cache_required(self) -> RequirementsBuilder:
        """Mark this operation as not requiring cache.

        Convenience method equivalent to `requires_cache(False)`.

        Returns:
            Self for method chaining.
        """
        self._requires_cache = False
        return self

    def use_cache_view(self, cache_view_class: type) -> RequirementsBuilder:
        """Set the cache view class this operation requires.

        The cache view class (e.g., TransformerCacheView, Mamba2CacheView)
        determines what type of cache will be initialized for this operation.

        Args:
            cache_view_class: The cache view class (not an instance).

        Returns:
            Self for method chaining.
        """
        self._cache_view_class = cache_view_class
        return self

    def build(self) -> OperationRequirements:
        """Build the OperationRequirements instance.

        Returns:
            The constructed OperationRequirements.
        """
        return OperationRequirements(
            metadata=MetadataRequirements(
                required=self._required_metadata,
                optional=self._optional_metadata,
            ),
            cache=CacheRequirements(
                supported=self._supported_cache,
                preferred=self._preferred_cache,
                requires_cache=self._requires_cache,
                cache_view_class=self._cache_view_class,
            ),
            name=self._name,
        )


class ModeSpecificBuilder:
    """Builder for mode-specific requirements.

    Example usage:
        >>> reqs = (ModeSpecificBuilder("flash_attention")
        ...     .for_prefill()
        ...         .require_metadata(MetadataField.QUERY_START_LOC)
        ...         .support_cache(CacheType.RAGGED_PAGES)
        ...     .for_decode()
        ...         .require_metadata(MetadataField.PAGES_TABLES)
        ...         .support_cache(CacheType.RAGGED_PAGES)
        ...     .build())
    """

    def __init__(self, name: str = ""):
        """Initialize the builder.

        Args:
            name: Name of the operation.
        """
        self._name = name
        self._prefill_builder = RequirementsBuilder(f"{name}_prefill")
        self._decode_builder = RequirementsBuilder(f"{name}_decode")
        self._mixed_builder: RequirementsBuilder | None = None
        self._current_builder: RequirementsBuilder = self._prefill_builder

    def for_prefill(self) -> ModeSpecificBuilder:
        """Switch to configuring prefill requirements.

        Returns:
            Self for method chaining.
        """
        self._current_builder = self._prefill_builder
        return self

    def for_decode(self) -> ModeSpecificBuilder:
        """Switch to configuring decode requirements.

        Returns:
            Self for method chaining.
        """
        self._current_builder = self._decode_builder
        return self

    def for_mixed(self) -> ModeSpecificBuilder:
        """Switch to configuring mixed mode requirements.

        If not explicitly configured, mixed mode will use union of prefill and decode.

        Returns:
            Self for method chaining.
        """
        if self._mixed_builder is None:
            self._mixed_builder = RequirementsBuilder(f"{self._name}_mixed")
        self._current_builder = self._mixed_builder
        return self

    def require_metadata(self, fields: MetadataField) -> ModeSpecificBuilder:
        """Add required metadata fields for current mode.

        Args:
            fields: Metadata fields that are required.

        Returns:
            Self for method chaining.
        """
        self._current_builder.require_metadata(fields)
        return self

    def optional_metadata(self, fields: MetadataField) -> ModeSpecificBuilder:
        """Add optional metadata fields for current mode.

        Args:
            fields: Metadata fields that are optional.

        Returns:
            Self for method chaining.
        """
        self._current_builder.optional_metadata(fields)
        return self

    def support_cache(self, cache_types: CacheType) -> ModeSpecificBuilder:
        """Set supported cache types for current mode.

        Args:
            cache_types: Cache types that are supported.

        Returns:
            Self for method chaining.
        """
        self._current_builder.support_cache(cache_types)
        return self

    def prefer_cache(self, cache_type: CacheType) -> ModeSpecificBuilder:
        """Set preferred cache type for current mode.

        Args:
            cache_type: The preferred cache type.

        Returns:
            Self for method chaining.
        """
        self._current_builder.prefer_cache(cache_type)
        return self

    def requires_cache(self, value: bool = True) -> ModeSpecificBuilder:
        """Set whether current mode requires cache.

        Args:
            value: True if operation requires cache, False otherwise.

        Returns:
            Self for method chaining.
        """
        self._current_builder.requires_cache(value)
        return self

    def no_cache_required(self) -> ModeSpecificBuilder:
        """Mark current mode as not requiring cache.

        Returns:
            Self for method chaining.
        """
        self._current_builder.no_cache_required()
        return self

    def use_cache_view(self, cache_view_class: type) -> ModeSpecificBuilder:
        """Set the cache view class for current mode.

        Args:
            cache_view_class: The cache view class (not an instance).

        Returns:
            Self for method chaining.
        """
        self._current_builder.use_cache_view(cache_view_class)
        return self

    def build(self) -> ModeSpecificRequirements:
        """Build the ModeSpecificRequirements instance.

        Returns:
            The constructed ModeSpecificRequirements.
        """
        return ModeSpecificRequirements(
            prefill=self._prefill_builder.build(),
            decode=self._decode_builder.build(),
            mixed=self._mixed_builder.build() if self._mixed_builder else None,
        )

    def get(self, mode: ExecutionMode) -> OperationRequirements:
        """Build and get requirements for a specific mode.

        Convenience method that builds and returns requirements for one mode.

        Args:
            mode: The execution mode.

        Returns:
            The requirements for that mode.
        """
        return self.build().get(mode)
