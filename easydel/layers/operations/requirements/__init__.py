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

"""Operation requirements system for declarative metadata and cache specifications.

This module provides a system for operations (attention mechanisms, etc.) to
declare their requirements in terms of:

1. **Metadata fields**: What runtime information the operation needs
   (sequence lengths, page tables, positions, etc.)

2. **Cache types**: What cache implementations the operation supports
   (transformer cache, ragged pages cache, recurrent cache, hybrid cache)

The inference engine uses these declarations to:
- Build only the required metadata (avoiding unnecessary computation)
- Validate cache compatibility at initialization
- Provide clear error messages when requirements aren't met

Example usage:

    >>> from easydel.layers.operations.requirements import (
    ...     RequirementsBuilder,
    ...     MetadataField,
    ...     CacheType,
    ...     ExecutionMode,
    ... )
    >>>
    >>> # Build requirements for a paged attention operation
    >>> reqs = (RequirementsBuilder("ragged_page_attention_v3")
    ...     .require_metadata(
    ...         MetadataField.QUERY_START_LOC |
    ...         MetadataField.PAGES_TABLES |
    ...         MetadataField.REQUEST_DISTRIBUTION
    ...     )
    ...     .optional_metadata(MetadataField.LOGITS_INDICES)
    ...     .support_cache(CacheType.RAGGED_PAGES)
    ...     .build())
    >>>
    >>> # Check what metadata is required
    >>> MetadataField.PAGES_TABLES in reqs.metadata.required
    True
"""

from .builder import ModeSpecificBuilder, RequirementsBuilder
from .requirements import (
    CacheRequirements,
    MetadataRequirements,
    ModeSpecificRequirements,
    OperationRequirements,
)
from .types import CacheType, ExecutionMode, MetadataField
from .validation import (
    RequirementsValidator,
    ValidationResult,
    get_metadata_field_names,
    validate_cache_compatibility,
    validate_metadata_availability,
)

__all__ = [
    "CacheRequirements",
    "CacheType",
    "ExecutionMode",
    "MetadataField",
    "MetadataRequirements",
    "ModeSpecificBuilder",
    "ModeSpecificRequirements",
    "OperationRequirements",
    "RequirementsBuilder",
    "RequirementsValidator",
    "ValidationResult",
    "get_metadata_field_names",
    "validate_cache_compatibility",
    "validate_metadata_availability",
]
