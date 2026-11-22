"""Operation requirements builders and validators (no array computation)."""

import pytest
from easydel.operations.requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    ModeSpecificBuilder,
    ModeSpecificRequirements,
    OperationRequirements,
    RequirementsBuilder,
    RequirementsValidator,
    get_metadata_field_names,
    validate_cache_compatibility,
    validate_metadata_availability,
)

pytestmark = pytest.mark.cpu_ok


def test_validator_reports_all_failures_and_optional_warnings():
    requirements = (
        RequirementsBuilder("paged")
        .support_cache(CacheType.RAGGED_PAGES)
        .require_metadata(MetadataField.PAGES_TABLES)
        .optional_metadata(MetadataField.LOGITS_INDICES)
        .build()
    )
    invalid = RequirementsValidator(CacheType.RECURRENT, MetadataField.NONE).validate(requirements)
    assert not invalid
    assert len(invalid.errors) == 2
    with pytest.raises(ValueError, match="paged"):
        invalid.raise_if_invalid()
    valid = RequirementsValidator(CacheType.RAGGED_PAGES, MetadataField.PAGES_TABLES).validate(requirements)
    assert valid
    assert valid.errors == []
    assert len(valid.warnings) == 1
    valid.raise_if_invalid()
    assert validate_cache_compatibility(requirements, CacheType.RAGGED_PAGES)
    assert not validate_metadata_availability(requirements, MetadataField.NONE)
    assert get_metadata_field_names(MetadataField.PAGES_TABLES | MetadataField.POSITIONS) == [
        "POSITIONS",
        "PAGES_TABLES",
    ]


def test_validator_combines_requirements_across_operations():
    first = OperationRequirements.create(
        name="first",
        required_metadata=MetadataField.POSITIONS,
        supported_cache=CacheType.TRANSFORMER | CacheType.RAGGED_PAGES,
    )
    second = OperationRequirements.create(
        name="second",
        required_metadata=MetadataField.PAGES_TABLES,
        supported_cache=CacheType.RAGGED_PAGES | CacheType.RECURRENT,
    )
    validator = RequirementsValidator(CacheType.RAGGED_PAGES, MetadataField.POSITIONS | MetadataField.PAGES_TABLES)
    assert validator.validate_all([first, second])
    assert validator.get_combined_metadata_requirements([first, second]) == (
        MetadataField.POSITIONS | MetadataField.PAGES_TABLES
    )
    assert validator.get_common_cache_types([first, second]) == CacheType.RAGGED_PAGES
    assert validator.get_common_cache_types([]) == CacheType.any()
    assert validator.validate_all([])


def test_mode_specific_builder_resolves_implicit_and_explicit_mixed_modes():
    builder = (
        ModeSpecificBuilder("attention")
        .for_prefill()
        .require_metadata(MetadataField.QUERY_START_LOC)
        .support_cache(CacheType.TRANSFORMER | CacheType.RAGGED_PAGES)
        .for_decode()
        .require_metadata(MetadataField.PAGES_TABLES)
        .support_cache(CacheType.RAGGED_PAGES)
    )
    requirements = builder.build()
    assert requirements.get(ExecutionMode.PREFILL).metadata.required == MetadataField.QUERY_START_LOC
    assert requirements.get(ExecutionMode.DECODE).metadata.required == MetadataField.PAGES_TABLES
    mixed = requirements.get(ExecutionMode.MIXED)
    assert mixed.metadata.required == MetadataField.QUERY_START_LOC | MetadataField.PAGES_TABLES
    assert mixed.cache.supported == CacheType.RAGGED_PAGES
    explicit = builder.for_mixed().require_metadata(MetadataField.REQUEST_DISTRIBUTION).no_cache_required().build()
    assert explicit.get(ExecutionMode.MIXED).metadata.required == MetadataField.REQUEST_DISTRIBUTION
    assert not explicit.get(ExecutionMode.MIXED).cache.requires_cache


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("cache_type", [CacheType.NONE, CacheType.RAGGED_PAGES, CacheType.RECURRENT])
def test_optional_cache_accepts_absence_but_rejects_incompatible_supplied_cache(required, cache_type):
    requirements = (
        RequirementsBuilder("optional-paged").support_cache(CacheType.RAGGED_PAGES).requires_cache(required).build()
    )
    expected = cache_type == CacheType.RAGGED_PAGES or (cache_type == CacheType.NONE and not required)
    assert bool(RequirementsValidator(cache_type, MetadataField.NONE).validate(requirements)) == expected
    assert bool(validate_cache_compatibility(requirements, cache_type)) == expected
    assert requirements.cache.is_compatible_with(cache_type) == expected


def test_no_cache_builder_validates_without_cache():
    requirements = RequirementsBuilder("no-cache").no_cache_required().build()
    assert RequirementsValidator(CacheType.NONE, MetadataField.NONE).validate(requirements)


def test_uniform_requirements_and_additive_cache_support():
    requirements = (
        RequirementsBuilder("hybrid").support_cache(CacheType.RECURRENT).add_cache_support(CacheType.HYBRID).build()
    )
    assert requirements.cache.supported == CacheType.RECURRENT | CacheType.HYBRID
    uniform = ModeSpecificRequirements.uniform(requirements)
    for mode in ExecutionMode:
        assert uniform.get(mode) == requirements
