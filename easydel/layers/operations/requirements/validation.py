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

"""Validation utilities for operation requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .requirements import OperationRequirements
from .types import CacheType, ExecutionMode, MetadataField

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "RequirementsValidator",
    "ValidationResult",
    "get_metadata_field_names",
    "validate_cache_compatibility",
    "validate_metadata_availability",
]


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        is_valid: Whether the validation passed.
        errors: List of error messages if validation failed.
        warnings: List of warning messages (non-fatal issues).
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]

    @classmethod
    def success(cls, warnings: list[str] | None = None) -> ValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True, errors=[], warnings=warnings or [])

    @classmethod
    def failure(cls, errors: list[str], warnings: list[str] | None = None) -> ValidationResult:
        """Create a failed validation result."""
        return cls(is_valid=False, errors=errors, warnings=warnings or [])

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid

    def raise_if_invalid(self, error_class: type[Exception] = ValueError) -> None:
        """Raise an exception if validation failed.

        Args:
            error_class: The exception class to raise.

        Raises:
            error_class: If validation failed, with all errors in the message.
        """
        if not self.is_valid:
            raise error_class("\n".join(self.errors))


def get_metadata_field_names(fields: MetadataField) -> list[str]:
    """Get human-readable names for metadata fields.

    Args:
        fields: The metadata fields to get names for.

    Returns:
        List of field names.
    """
    names = []
    for field in MetadataField:
        if field != MetadataField.NONE and field in fields:
            names.append(field.name)
    return names


def validate_cache_compatibility(
    requirements: OperationRequirements,
    cache_type: CacheType,
) -> ValidationResult:
    """Validate that a cache type is compatible with operation requirements.

    Args:
        requirements: The operation requirements.
        cache_type: The cache type to validate.

    Returns:
        ValidationResult indicating compatibility.
    """
    if requirements.cache.is_compatible_with(cache_type):
        return ValidationResult.success()

    supported_names = [ct.name for ct in CacheType if ct in requirements.cache.supported and ct != CacheType.NONE]
    return ValidationResult.failure(
        errors=[
            f"Operation '{requirements.name}' does not support cache type '{cache_type.name}'. "
            f"Supported types: {supported_names}"
        ]
    )


def validate_metadata_availability(
    requirements: OperationRequirements,
    available: MetadataField,
) -> ValidationResult:
    """Validate that all required metadata fields are available.

    Args:
        requirements: The operation requirements.
        available: The metadata fields that are available.

    Returns:
        ValidationResult indicating whether all required fields are present.
    """
    if requirements.metadata.is_satisfied_by(available):
        # Check for unused optional fields
        unused_optional = requirements.metadata.optional & ~available
        warnings = []
        if unused_optional != MetadataField.NONE:
            unused_names = get_metadata_field_names(unused_optional)
            warnings.append(f"Optional metadata fields not available: {unused_names}")
        return ValidationResult.success(warnings=warnings)

    missing = requirements.metadata.missing_fields(available)
    missing_names = get_metadata_field_names(missing)
    return ValidationResult.failure(
        errors=[f"Operation '{requirements.name}' requires metadata fields that are not available: {missing_names}"]
    )


class RequirementsValidator:
    """Validator for operation requirements against runtime configuration.

    Provides comprehensive validation of operation requirements including
    cache compatibility and metadata availability.
    """

    def __init__(
        self,
        cache_type: CacheType,
        available_metadata: MetadataField,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ):
        """Initialize the validator.

        Args:
            cache_type: The cache type being used.
            available_metadata: The metadata fields that are available.
            mode: The execution mode.
        """
        self.cache_type = cache_type
        self.available_metadata = available_metadata
        self.mode = mode

    def validate(self, requirements: OperationRequirements) -> ValidationResult:
        """Validate operation requirements against runtime configuration.

        Args:
            requirements: The operation requirements to validate.

        Returns:
            ValidationResult with all validation errors and warnings.
        """
        all_errors: list[str] = []
        all_warnings: list[str] = []

        # Validate cache compatibility
        cache_result = validate_cache_compatibility(requirements, self.cache_type)
        all_errors.extend(cache_result.errors)
        all_warnings.extend(cache_result.warnings)

        # Validate metadata availability
        metadata_result = validate_metadata_availability(requirements, self.available_metadata)
        all_errors.extend(metadata_result.errors)
        all_warnings.extend(metadata_result.warnings)

        if all_errors:
            return ValidationResult.failure(errors=all_errors, warnings=all_warnings)
        return ValidationResult.success(warnings=all_warnings)

    def validate_all(
        self,
        requirements_list: Sequence[OperationRequirements],
    ) -> ValidationResult:
        """Validate multiple operation requirements.

        Args:
            requirements_list: List of operation requirements to validate.

        Returns:
            Combined ValidationResult for all operations.
        """
        all_errors: list[str] = []
        all_warnings: list[str] = []

        for reqs in requirements_list:
            result = self.validate(reqs)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        if all_errors:
            return ValidationResult.failure(errors=all_errors, warnings=all_warnings)
        return ValidationResult.success(warnings=all_warnings)

    def get_combined_metadata_requirements(
        self,
        requirements_list: Sequence[OperationRequirements],
    ) -> MetadataField:
        """Get the union of all required metadata fields.

        Args:
            requirements_list: List of operation requirements.

        Returns:
            Combined required metadata fields.
        """
        combined = MetadataField.NONE
        for reqs in requirements_list:
            combined |= reqs.metadata.required
        return combined

    def get_common_cache_types(
        self,
        requirements_list: Sequence[OperationRequirements],
    ) -> CacheType:
        """Get cache types supported by all operations.

        Args:
            requirements_list: List of operation requirements.

        Returns:
            Cache types that are supported by all operations.
        """
        if not requirements_list:
            return CacheType.any()

        common = requirements_list[0].cache.supported
        for reqs in requirements_list[1:]:
            common &= reqs.cache.supported
        return common
