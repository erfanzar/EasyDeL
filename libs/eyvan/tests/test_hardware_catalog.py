# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One source of shape truth, including provider-sourced host resources.

``fleet_config`` held a compute-TPU dict and a queued-slice dict;
``autoscale.profiles`` derived a chip count from the type suffix and hardcoded
``cpu=32, memory_gib=0`` for every TPU; every fleet YAML restated part of the
answer by hand. A discovered type name never supplied RAM, so that fallback
created durable pools which no default workload could use.

The resource literals below come from the versioned Google Cloud pages cited by
each catalog row, not from production code. Consumption tests then follow the
value through autoscale, fleet/API compilation, durable records, and cells so a
field that exists only in the YAML cannot produce a false pass.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from eyvan.capacity.autoscale.profiles import DEFAULT_TPU_PROFILES, tpu_profile_from_accelerator_type
from eyvan.capacity.hardware_catalog import (
    CATALOG_PATH,
    SUPPORTED_CATALOG_VERSION,
    GenerationRule,
    HardwareCatalogError,
    hardware_catalog,
    load_hardware_catalog,
    shape_for,
)

#: Provider-published per-TPU-VM resources, written as independent literals.
#: Google Cloud's v5p and v6e pages establish these CPU/RAM values and
#: topology-dependent VM shapes.
EXPECTED = {
    "v5p-8": {"chips": 4, "quota_units": 8, "hosts": 1, "cpu": 208, "memory_gib": 448, "source": "v5p"},
    "v5p-16": {"chips": 8, "quota_units": 16, "hosts": 2, "cpu": 208, "memory_gib": 448, "source": "v5p"},
    "v5p-32": {"chips": 16, "quota_units": 32, "hosts": 4, "cpu": 208, "memory_gib": 448, "source": "v5p"},
    "v6e-8": {"chips": 8, "quota_units": 8, "hosts": None, "cpu": 360, "memory_gib": 1440, "source": "v6e"},
    "v6e-16": {"chips": 16, "quota_units": 16, "hosts": None, "cpu": 180, "memory_gib": 720, "source": "v6e"},
}


def _catalog_document() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, document: dict):
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("accelerator_type", sorted(EXPECTED))
def test_the_catalog_states_the_shape_the_fleet_actually_runs(accelerator_type: str) -> None:
    """Literals, not a comparison between two things that could both be wrong."""

    expected = EXPECTED[accelerator_type]
    shape = hardware_catalog().require(accelerator_type)

    assert shape.chips == expected["chips"]
    assert shape.quota_units == expected["quota_units"]
    assert shape.hosts == expected["hosts"]
    assert shape.host_cpu == expected["cpu"]
    assert shape.host_memory_gib == expected["memory_gib"]
    assert shape.source_url == f"https://cloud.google.com/tpu/docs/{expected['source']}"
    assert shape.source_version == "Last updated 2026-08-04 UTC"


#: Exactly what the two deleted dictionaries in ``fleet_config`` contained. Kept
#: as literals so retiring them cannot quietly change what a pool is provisioned
#: with -- the whole risk of a consolidation is that the survivor drifts.
RETIRED_COMPUTE_PROFILE = {
    "v5p-8": {
        "machine_type": "ct5p-hightpu-4t",
        "tpu_type": "ct5p",
        "tpu_count": 4,
        "image_family": "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
        "image_project": "ubuntu-os-accelerator-images",
    },
}
RETIRED_QUEUED_SLICE_PROFILE = {"v5p-16": 8, "v5p-32": 16}

#: Names and TPU counts the old compatibility tuple exposed. Its guessed CPU
#: value is intentionally not retained now that provider-sourced values exist.
RETIRED_DEFAULT_PROFILES = (("v5p-16", 8), ("v5p-32", 16), ("v6e-8", 8))


def test_the_catalog_carries_what_the_compute_dictionary_carried() -> None:
    """``v5p-8`` is provisioned as an integrated Compute Engine TPU VM.

    Its machine type, image and chip count used to live in a second dictionary
    that had to agree with everything else. Losing any of these on the way to
    the catalog would provision the wrong machine.
    """

    for accelerator_type, profile in RETIRED_COMPUTE_PROFILE.items():
        shape = hardware_catalog().require(accelerator_type)
        assert shape.chips == profile["tpu_count"]
        assert shape.machine_type == profile["machine_type"]
        assert shape.tpu_type == profile["tpu_type"]
        assert shape.image_family == profile["image_family"]
        assert shape.image_project == profile["image_project"]


def test_the_compiled_provider_options_still_carry_the_compute_profile() -> None:
    """The values have to survive the whole path, not merely exist in the catalog.

    ``compiled_provider_options`` is what actually reaches the provider, and it
    used to spread the dictionary in wholesale.
    """

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    config = EyvanFleetConfig.model_validate(
        _fleet({"min_workers": 0, "max_workers": 1, "capacity_backend": "gcp-compute"}, name="v5p-8")
    )
    (pool,) = config.pool_payloads()
    options = pool["provider_options"]
    expected = RETIRED_COMPUTE_PROFILE["v5p-8"]

    assert options["machine_type"] == expected["machine_type"]
    assert options["tpu_type"] == expected["tpu_type"]
    assert options["tpu_count"] == expected["tpu_count"]
    assert options["image_family"] == expected["image_family"]
    assert options["image_project"] == expected["image_project"]


def test_the_catalog_carries_what_the_queued_slice_dictionary_carried() -> None:
    """Each queued worker is one independent slice, and its chip count is its size."""

    for accelerator_type, chips in RETIRED_QUEUED_SLICE_PROFILE.items():
        assert hardware_catalog().require(accelerator_type).chips == chips


def test_the_derived_compatibility_tuple_still_contains_what_it_used_to() -> None:
    """Deriving it must not silently drop an entry someone imports.

    ``v6e-8`` was in the hand-written tuple and nowhere else, which is exactly
    how a fourth copy holds the only record of something.
    """

    derived = {(profile.accelerator_type, profile.tpu) for profile in DEFAULT_TPU_PROFILES}

    assert set(RETIRED_DEFAULT_PROFILES) <= derived
    assert all(profile.cpu > 0 and profile.memory_gib > 0 for profile in DEFAULT_TPU_PROFILES)
    # v5p-8 is compiled as an integrated Compute Engine TPU VM and excluded from
    # queued autoscaling, so offering a profile for it would be offering one the
    # autoscaler refuses to use.
    assert "v5p-8" not in {profile.accelerator_type for profile in DEFAULT_TPU_PROFILES}


def test_the_catalog_agrees_with_the_derived_autoscale_profile() -> None:
    """The suffix-derived path is what runs today; the catalog must not move it.

    ``profiles`` divides the type suffix by two for v4 and v5p and by one
    otherwise. Reproducing a different chip count would change how much hardware
    an autoscaled pool asks for.
    """

    for accelerator_type in EXPECTED:
        derived = tpu_profile_from_accelerator_type(accelerator_type)
        if derived is None:
            # v5p-8 is deliberately excluded from queued autoscaling.
            assert accelerator_type == "v5p-8"
            continue
        shape = hardware_catalog().require(accelerator_type)
        assert shape.schedulable_tpu == derived.tpu, accelerator_type
        assert shape.host_cpu == derived.cpu, accelerator_type
        assert shape.host_memory_gib == derived.memory_gib, accelerator_type


def test_the_catalog_agrees_with_the_compatibility_profile_tuple() -> None:
    """``DEFAULT_TPU_PROFILES`` is the fourth copy, and it is still consulted."""

    for profile in DEFAULT_TPU_PROFILES:
        shape = shape_for(profile.accelerator_type)
        if shape is None:
            continue
        assert (shape.schedulable_tpu, shape.host_cpu, shape.host_memory_gib) == (
            profile.tpu,
            profile.cpu,
            profile.memory_gib,
        ), profile.accelerator_type


# --------------------------------------------------------------------------- invariants


def test_the_shipped_catalog_loads() -> None:
    """The one test that fails on a bad edit before anything else runs."""

    catalog = hardware_catalog()

    assert catalog.catalog_version == SUPPORTED_CATALOG_VERSION
    assert catalog.revision
    assert catalog.accelerator_types


def test_a_chip_count_that_contradicts_its_topology_is_refused(tmp_path) -> None:
    """``chips == hosts * chips_per_host`` turns a typo into a load failure.

    Without it, a mistyped host count reaches a provider as a request for the
    wrong number of machines, and the first symptom is a JAX cluster that never
    forms.
    """

    document = _catalog_document()
    document["shapes"][1]["hosts"] = 3

    with pytest.raises(HardwareCatalogError, match="is not 3 hosts"):
        load_hardware_catalog(_write(tmp_path, document))


def test_quota_units_must_match_the_name_the_provider_uses(tmp_path) -> None:
    """The suffix *is* the quota unit count; a catalog disagreeing with it is wrong.

    Getting this backwards is how a v5p-16 pool ends up requesting twice the
    hardware -- chips and quota units differ by two on this generation.
    """

    document = _catalog_document()
    document["shapes"][1]["quota_units"] = 8

    with pytest.raises(HardwareCatalogError, match="its name says 16"):
        load_hardware_catalog(_write(tmp_path, document))


def test_a_generation_that_contradicts_the_name_is_refused(tmp_path) -> None:
    """Generation drives the quota divisor, so a wrong one mis-sizes every pool."""

    document = _catalog_document()
    document["shapes"][1]["generation"] = "v6e"

    with pytest.raises(HardwareCatalogError, match="its name says 'v5p'"):
        load_hardware_catalog(_write(tmp_path, document))


def test_half_a_topology_is_refused(tmp_path) -> None:
    """Hosts without chips-per-host cannot be checked, so it must not be storable.

    A half-recorded topology reads as knowledge and is not: it would pass the
    product invariant by skipping it.
    """

    document = _catalog_document()
    document["shapes"][3]["hosts"] = 4

    with pytest.raises(HardwareCatalogError, match="recorded together or not at all"):
        load_hardware_catalog(_write(tmp_path, document))


def test_a_duplicate_type_is_refused(tmp_path) -> None:
    """Two entries for one type is the failure this catalog exists to end.

    It must not be reintroducible inside the catalog itself, where it would be
    even harder to notice than across three files.
    """

    document = _catalog_document()
    document["shapes"].append(dict(document["shapes"][1]))

    with pytest.raises(HardwareCatalogError, match="more than once"):
        load_hardware_catalog(_write(tmp_path, document))


def test_an_unsupported_schema_version_is_refused_rather_than_partly_read(tmp_path) -> None:
    """A build that ignored unknown fields would size pools from half a description."""

    document = _catalog_document()
    document["catalog_version"] = 99

    with pytest.raises(HardwareCatalogError, match=f"this build understands {SUPPORTED_CATALOG_VERSION}"):
        load_hardware_catalog(_write(tmp_path, document))


def test_a_fractional_host_cpu_is_refused(tmp_path) -> None:
    """A host has whole CPUs, and ``TpuProfile`` requires a positive integer.

    Storing a float would move the coercion to every call site, where one of
    them would eventually truncate silently.
    """

    document = _catalog_document()
    document["shapes"][1]["host_cpu"] = 31.5

    with pytest.raises(HardwareCatalogError, match="positive integer"):
        load_hardware_catalog(_write(tmp_path, document))


@pytest.mark.parametrize("value", [0, -1, float("inf")])
def test_non_positive_or_non_finite_host_memory_is_refused(tmp_path, value: float) -> None:
    """Admission memory cannot be absent, zero, negative, or non-finite."""

    document = _catalog_document()
    document["shapes"][1]["host_memory_gib"] = value

    with pytest.raises(HardwareCatalogError, match="host_memory_gib must be a positive finite number"):
        load_hardware_catalog(_write(tmp_path, document))


def test_every_accelerator_entry_requires_versioned_provider_provenance(tmp_path) -> None:
    """Future TPU or GPU rows cannot turn an operator's estimate into catalog fact."""

    document = _catalog_document()
    document["shapes"][1].pop("source_version")

    with pytest.raises(HardwareCatalogError, match="source_version"):
        load_hardware_catalog(_write(tmp_path, document))

    document = _catalog_document()
    document["shapes"][1]["source_url"] = "provider-notes.txt"
    with pytest.raises(HardwareCatalogError, match="absolute HTTPS URL"):
        load_hardware_catalog(_write(tmp_path, document))


def test_a_catalog_without_a_revision_is_refused(tmp_path) -> None:
    """The revision is how an operator tells which numbers a pool was sized from."""

    document = _catalog_document()
    document.pop("revision")

    with pytest.raises(HardwareCatalogError, match="non-empty revision"):
        load_hardware_catalog(_write(tmp_path, document))


def test_malformed_yaml_names_the_file(tmp_path) -> None:
    """A parse error with no path is a support ticket."""

    path = tmp_path / "catalog.yaml"
    path.write_text(
        textwrap.dedent("""\
        catalog_version: 3
        shapes: [
        """),
        encoding="utf-8",
    )

    with pytest.raises(HardwareCatalogError, match="not valid YAML"):
        load_hardware_catalog(path)


#: A real GCP accelerator type this catalog deliberately describes no shape for.
#:
#: It has to be real -- a made-up name would exercise the parser's rejection path
#: rather than the "known provider type, unknown host resources" path these tests
#: are about. v4 has no explicit row and no generation rule, so nothing here
#: establishes its host CPU or memory. If a v4 rule is ever added, every test
#: using this fails loudly and asks for a new one, which is the intent.
UNCATALOGUED_TYPE = "v4-32"


def test_the_uncatalogued_fixture_is_still_uncatalogued() -> None:
    """Guards the three tests below, whose whole subject is an unknown type."""

    assert shape_for(UNCATALOGUED_TYPE) is None


def test_an_unknown_type_is_refused_with_the_alternatives(tmp_path) -> None:
    """An error an operator can act on names what the catalog does know."""

    with pytest.raises(HardwareCatalogError) as raised:
        hardware_catalog().require(UNCATALOGUED_TYPE)

    message = str(raised.value)
    assert UNCATALOGUED_TYPE in message
    assert "v5p-16" in message, "the refusal must list the types that are available"


def test_chips_and_quota_units_are_kept_distinct() -> None:
    """The distinction the old code blurred, on the generation where it matters.

    v5p bills two quota units per chip. A caller reading ``quota_units`` as a
    schedulable TPU count would double every request.
    """

    shape = hardware_catalog().require("v5p-16")

    assert (shape.chips, shape.quota_units) == (8, 16)
    assert shape.schedulable_tpu == 8


def test_every_shipped_shape_states_the_host_it_describes() -> None:
    """The invariant, applied to every kind rather than to the rows that existed.

    ``EXPECTED`` above names TPU rows one by one; a GPU row added without a host
    CPU or memory figure would simply not be listed there and would ship. This
    is the check that does not need updating when a kind is added.
    """

    for shape in hardware_catalog().shapes:
        assert shape.kind in {"gpu", "tpu"}, shape.accelerator_type
        assert shape.host_cpu > 0, shape.accelerator_type
        assert shape.host_memory_gib > 0, shape.accelerator_type
        assert shape.source_url.startswith("https://"), shape.accelerator_type
        assert shape.source_version.strip(), shape.accelerator_type
        # Exactly one dimension, so a caller adding shapes up cannot double-count.
        assert (shape.schedulable_tpu > 0) != (shape.schedulable_gpu > 0), shape.accelerator_type


def test_a_gpu_shape_never_becomes_a_queued_tpu_autoscale_worker() -> None:
    """Queued resources are a Cloud TPU mechanism; a GPU row must not reach it.

    ``DEFAULT_TPU_PROFILES`` and the discovery path both read the whole catalog,
    so adding a kind to the catalog is exactly when a filter that was previously
    unobservable starts to matter.
    """

    gpu_types = {shape.accelerator_type for shape in hardware_catalog().shapes if shape.kind == "gpu"}

    assert gpu_types, "the catalog no longer ships a GPU row and this test is checking nothing"
    assert gpu_types.isdisjoint({profile.accelerator_type for profile in DEFAULT_TPU_PROFILES})
    for accelerator_type in gpu_types:
        assert tpu_profile_from_accelerator_type(accelerator_type) is None, accelerator_type


def test_the_catalog_is_packaged_in_the_wheel() -> None:
    """Data beside code is only shipped if the build backend is told.

    A catalog that resolves in a source checkout and vanishes in the container
    would fail at the first pool upsert on a real deployment.
    """

    assert CATALOG_PATH.is_file()
    assert CATALOG_PATH.parent.name == "data"
    assert CATALOG_PATH.parent.parent.name == "capacity"


# --------------------------------------------------------------------------- consumption


def _fleet(pool: dict, name: str = "v5p-16") -> dict:
    return {
        "version": 1,
        "name": "research-fleet",
        "provider": {"type": "gcp", "project": "p", "region": "us-east5", "zone": "us-east5-a"},
        "executor": "eyvan-native",
        "capacity_class": "spot",
        "agent_endpoint": "https://agent.example.com",
        "pools": {name: {**pool, "provider_options": {"bootstrap_egress": "public-internet"}}},
    }


def test_a_pool_named_after_a_known_shape_needs_no_resources_block() -> None:
    """The fourth copy, deleted. Every fleet YAML restated what the key already said.

    An operator writing ``v5p-16:`` has stated the shape; making them also write
    ``TPU: 8`` underneath is asking them to agree with a number they have no way
    to look up.
    """

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    config = EyvanFleetConfig.model_validate(_fleet({"min_workers": 0, "max_workers": 1}))
    (pool,) = config.pool_payloads()

    assert pool["capacity"]["tpu"] == 8
    assert pool["capacity"]["cpu"] == 208
    assert pool["capacity"]["memory_gib"] == 448
    assert pool["allocatable"]["memory_gib"] == 448


@pytest.mark.parametrize("accelerator_type", sorted(EXPECTED))
def test_every_catalog_shape_reaches_a_durable_schedulable_pool(accelerator_type: str) -> None:
    """Prove memory survives catalog -> fleet -> API record -> cell/document.

    Merely asserting the YAML field would miss the original defect: autoscale and
    durable pool compilation each used their own resource vector. The default
    workload asks for one CPU and one GiB plus the complete shape TPU count.
    """

    from eyvan._internal.json_types import model_json
    from eyvan.capacity.fleet_config import EyvanFleetConfig
    from eyvan.records import CapacityPoolRecord, ResourceVector
    from eyvan.server.api.schemas import FleetApplyRequest

    shape = hardware_catalog().require(accelerator_type)
    config = EyvanFleetConfig.model_validate(_fleet({"min_workers": 0, "max_workers": 1}, name=accelerator_type))
    (payload,) = config.pool_payloads()
    request = FleetApplyRequest.model_validate({"pools": [payload]})
    (record,) = request.to_records(default_agent_endpoint="https://agent.example.com")
    restored = CapacityPoolRecord.model_validate(model_json(record))
    cell = restored.make_cell("catalog-cell")
    default_workload = ResourceVector(tpu=shape.schedulable_tpu)

    assert restored.capacity.memory_gib == shape.host_memory_gib
    assert restored.allocatable.memory_gib == shape.host_memory_gib
    assert cell.capacity.memory_gib == shape.host_memory_gib
    assert default_workload.cpu <= cell.allocatable.cpu
    assert default_workload.memory_gib <= cell.allocatable.memory_gib
    assert default_workload.tpu <= cell.allocatable.tpu


def test_a_declared_block_that_contradicts_the_catalog_is_refused() -> None:
    """Client-side, before any API call, naming the number that is wrong.

    This is the check that catches a hand-written mistake -- and it is now one
    check instead of two dictionary lookups covering different subsets of types.
    """

    from pydantic import ValidationError

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    with pytest.raises(ValidationError, match=r"resources\.tpu == 8"):
        EyvanFleetConfig.model_validate(_fleet({"min_workers": 0, "max_workers": 1, "resources": {"CPU": 32, "TPU": 4}}))


def test_a_matching_declared_block_still_works() -> None:
    """Every fleet YAML in this repository declares one, and none may break."""

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    config = EyvanFleetConfig.model_validate(
        _fleet({"min_workers": 0, "max_workers": 1, "resources": {"CPU": 32, "TPU": 8}})
    )
    (pool,) = config.pool_payloads()

    assert pool["capacity"]["tpu"] == 8
    assert pool["capacity"]["cpu"] == 32, "an explicit positive schedulable override must be retained"
    assert pool["capacity"]["memory_gib"] == 448, "an omitted dimension must come from the catalog"


def test_a_known_pool_cannot_explicitly_restore_zero_memory() -> None:
    """A stale override must fail before it can recreate the legacy durable row."""

    from pydantic import ValidationError

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    with pytest.raises(ValidationError, match=r"resources\.memory_gib > 0"):
        EyvanFleetConfig.model_validate(
            _fleet(
                {
                    "min_workers": 0,
                    "max_workers": 1,
                    "resources": {"CPU": 32, "memory_gib": 0, "TPU": 8},
                }
            )
        )


def test_api_rejects_new_zero_memory_pool_but_legacy_record_remains_readable() -> None:
    """Write-time safety must not hide the rows doctor needs operators to repair."""

    from pydantic import ValidationError

    from eyvan.capacity.fleet_config import EyvanFleetConfig
    from eyvan.records import CapacityPoolRecord
    from eyvan.server.api.schemas import FleetApplyRequest

    config = EyvanFleetConfig.model_validate(_fleet({"min_workers": 0, "max_workers": 1}))
    (payload,) = config.pool_payloads()
    payload["capacity"] = {"cpu": 32, "memory_gib": 0, "tpu": 8}
    payload["allocatable"] = {"cpu": 32, "memory_gib": 0, "tpu": 8}

    with pytest.raises(ValidationError, match=r"accelerator pool capacity requires positive cpu and memory_gib"):
        FleetApplyRequest.model_validate({"pools": [payload]})

    payload["agent_endpoint"] = "https://agent.example.com"
    legacy = CapacityPoolRecord.model_validate(payload)
    assert legacy.capacity.memory_gib == 0


def _direct_accelerator_write_payload(boundary: str) -> dict:
    common = {
        "provider": "gcp",
        "capacity_backend": "gcp-tpu",
        "provider_project": "gcp-project",
        "region": "us-east5",
        "zone": "us-east5-a",
        "capacity": {"cpu": 64, "memory_gib": 256, "tpu": 8},
        "allocatable": {"cpu": 32, "memory_gib": 128, "tpu": 8},
        "accelerator_type": "v5p-16",
        "agent_endpoint": "http://agent:8090",
    }
    if boundary == "pool":
        return {
            **common,
            "id": "research-v5p-16",
            "fleet_name": "research",
            "node_type": "v5p-16",
        }
    common.pop("agent_endpoint")
    return {**common, "id": "research-v5p-16-0001", "endpoint": "http://worker:8100"}


@pytest.mark.parametrize("boundary", ["pool", "cell"])
@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("catalog-kind", r"is a tpu type.*advertise gpu"),
        ("capacity-count", r"capacity\.tpu.*catalog count 8"),
        ("allocatable-count", r"allocatable\.tpu.*catalog count 8"),
        ("backend-kind", r"capacity_backend.*requires tpu resources"),
    ],
)
def test_direct_accelerator_write_boundaries_reject_impossible_gcp_semantics(
    boundary: str,
    invalid: str,
    message: str,
) -> None:
    """Direct pool and cell APIs must not bypass the typed fleet catalog guard."""

    from pydantic import ValidationError

    from eyvan.server.api.schemas import CapacityPoolUpsertRequest, CellUpsertRequest

    payload = _direct_accelerator_write_payload(boundary)
    if invalid == "catalog-kind":
        payload["capacity"] = {"cpu": 64, "memory_gib": 256, "gpu": 1}
        payload["allocatable"] = {"cpu": 32, "memory_gib": 128, "gpu": 1}
    elif invalid == "capacity-count":
        payload["capacity"] = {"cpu": 64, "memory_gib": 256, "tpu": 999}
    elif invalid == "allocatable-count":
        payload["allocatable"] = {"cpu": 32, "memory_gib": 128, "tpu": 4}
    else:
        payload["accelerator_type"] = "nvidia-l4"
        payload["capacity"] = {"cpu": 64, "memory_gib": 256, "gpu": 1}
        payload["allocatable"] = {"cpu": 32, "memory_gib": 128, "gpu": 1}
    request_type = CapacityPoolUpsertRequest if boundary == "pool" else CellUpsertRequest

    with pytest.raises(ValidationError, match=message):
        request_type.model_validate(payload)


@pytest.mark.parametrize("boundary", ["pool", "cell"])
def test_direct_known_shape_writes_allow_positive_cpu_and_memory_overrides(boundary: str) -> None:
    """The catalog fixes accelerator count, not an operator's positive host reservation."""

    from eyvan.server.api.schemas import CapacityPoolUpsertRequest, CellUpsertRequest

    payload = _direct_accelerator_write_payload(boundary)
    request_type = CapacityPoolUpsertRequest if boundary == "pool" else CellUpsertRequest

    request = request_type.model_validate(payload)

    assert request.capacity.cpu == 64
    assert request.capacity.memory_gib == 256
    assert request.allocatable.cpu == 32
    assert request.allocatable.memory_gib == 128
    assert request.capacity.tpu == request.allocatable.tpu == 8


@pytest.mark.parametrize("boundary", ["pool", "cell"])
def test_direct_gcp_backends_cannot_hide_behind_a_non_gcp_provider(boundary: str) -> None:
    """The driver follows ``capacity_backend``; the provider label cannot bypass its catalog."""

    from pydantic import ValidationError

    from eyvan.server.api.schemas import CapacityPoolUpsertRequest, CellUpsertRequest

    payload = _direct_accelerator_write_payload(boundary)
    payload["provider"] = "local"
    payload["capacity"]["tpu"] = 999
    payload["allocatable"]["tpu"] = 999
    request_type = CapacityPoolUpsertRequest if boundary == "pool" else CellUpsertRequest

    with pytest.raises(ValidationError, match=r"capacity_backend .* requires provider 'gcp'"):
        request_type.model_validate(payload)


def test_typed_fleet_gcp_backend_cannot_hide_behind_a_local_provider() -> None:
    """The declarative compiler shares the same provider/backend boundary as direct writes."""

    from pydantic import ValidationError

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    document = {
        "version": 1,
        "name": "local-fleet",
        "provider": {"type": "local", "region": "local"},
        "agent_endpoint": "https://agent.example.com",
        "executor": "eyvan-native",
        "node_types": {
            "gpu": {
                "kind": "gpu",
                "machine_type": "local-gpu",
                "accelerator_type": "nvidia-l4",
                "capacity_backend": "gcp-tpu",
                "resources": {"cpu": 8, "memory_gib": 32, "gpu": 1},
            }
        },
    }

    with pytest.raises(ValidationError, match=r"capacity_backend .* requires provider 'gcp'"):
        EyvanFleetConfig.model_validate(document)


def test_direct_unknown_gcp_shape_requires_and_accepts_the_complete_manual_vector() -> None:
    """An uncatalogued provider name is usable, but never with guessed host resources."""

    from pydantic import ValidationError

    from eyvan.server.api.schemas import CapacityPoolUpsertRequest

    incomplete = _direct_accelerator_write_payload("pool")
    incomplete["accelerator_type"] = UNCATALOGUED_TYPE
    incomplete["capacity"] = {"cpu": 64, "memory_gib": 256, "tpu": 32}
    incomplete["allocatable"] = {"cpu": 64, "memory_gib": 256}

    with pytest.raises(ValidationError, match=r"unknown GCP accelerator shape.*complete manual"):
        CapacityPoolUpsertRequest.model_validate(incomplete)

    incomplete["allocatable"]["tpu"] = 32
    request = CapacityPoolUpsertRequest.model_validate(incomplete)
    assert request.capacity.tpu == request.allocatable.tpu == 32


def test_a_pool_whose_shape_the_catalog_does_not_know_allows_complete_manual_resources() -> None:
    """The catalog is not a hardware gate, but unknown resources are operator-owned.

    Provider discovery supplies a type name, not host RAM. An explicit complete
    vector is therefore the intentional escape hatch instead of a guessed
    catalog entry or a zero-memory pool.
    """

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    config = EyvanFleetConfig.model_validate(
        _fleet(
            {"min_workers": 0, "max_workers": 1, "resources": {"CPU": 64, "memory_gib": 256, "TPU": 32}},
            name="v5p-64",
        )
    )
    (pool,) = config.pool_payloads()

    assert pool["capacity"] == {"cpu": 64.0, "memory_gib": 256.0, "gpu": 0, "tpu": 32, "custom": {}}


def test_an_unknown_pool_without_explicit_memory_is_refused() -> None:
    """A type suffix cannot silently manufacture schedulable host resources."""

    from pydantic import ValidationError

    from eyvan.capacity.fleet_config import EyvanFleetConfig

    with pytest.raises(ValidationError, match=r"unknown TPU shape.*explicit positive CPU and memory_gib"):
        EyvanFleetConfig.model_validate(
            _fleet({"min_workers": 0, "max_workers": 1, "resources": {"CPU": 64, "TPU": 32}}, name=UNCATALOGUED_TYPE)
        )


# --------------------------------------------------------------------------- host counts


def _tpu_runner(nodes: dict[str, tuple[str, ...]], accelerator_type: str):
    """A gcloud stand-in returning one node description per allocation."""

    import json

    from test_gcp_provider import _multislice_qr, result

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(_multislice_qr(accelerator_type)))
        node_id = args[args.index("describe") + 1]
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": accelerator_type,
                    "metadata": {"eyvan-uid": "0123456789abcdef0123456789abcdef"},
                    "networkEndpoints": [{"ipAddress": ip} for ip in nodes[node_id]],
                }
            )
        )

    return runner


def test_a_slice_short_of_hosts_is_not_reported_healthy() -> None:
    """The defect this property exists for: a silently half-sized JAX cluster.

    The provider response alone cannot catch it, because a slice missing a host
    looks exactly like a smaller slice. The run starts, forms a smaller mesh
    than it asked for, and the only symptom is throughput nobody can explain.
    """

    from test_gcp_provider import _multislice_record

    from eyvan.capacity import CapacityStatus
    from eyvan.capacity.providers.gcp.queued_resources import GcloudTpuProvider

    short = {
        "cell-a-r1-node-0": ("10.2.0.4", "10.2.0.5"),
        "cell-a-r1-node-1": ("10.2.0.6",),
    }

    observation = GcloudTpuProvider(runner=_tpu_runner(short, "v5p-16")).ensure(_multislice_record())

    assert observation.state is CapacityStatus.PROVISIONING
    assert "2 hosts per slice" in observation.detail
    assert "4 are expected" in observation.detail


def test_a_complete_slice_is_healthy() -> None:
    """The check must not block the shape it exists to require."""

    from test_gcp_provider import _multislice_record

    from eyvan.capacity import CapacityStatus
    from eyvan.capacity.providers.gcp.queued_resources import GcloudTpuProvider

    complete = {
        "cell-a-r1-node-0": ("10.2.0.4", "10.2.0.5"),
        "cell-a-r1-node-1": ("10.2.0.6", "10.2.0.7"),
    }

    observation = GcloudTpuProvider(runner=_tpu_runner(complete, "v5p-16")).ensure(_multislice_record())

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.observed_tpu_vms == 4


def test_a_shape_with_no_recorded_host_count_keeps_the_provider_authoritative() -> None:
    """An unknown host count must not become an outage.

    The catalog records topology only where this repository grounds it, so for
    everything else the behaviour is exactly what it was: the provider decides.
    Refusing here would make Eyvan reject slices it simply has not been taught.
    """

    from test_gcp_provider import _multislice_record

    from eyvan.capacity import CapacityStatus
    from eyvan.capacity.providers.gcp.queued_resources import GcloudTpuProvider

    assert hardware_catalog().require("v6e-16").hosts is None
    uneven = {
        "cell-a-r1-node-0": ("10.2.0.4", "10.2.0.5"),
        "cell-a-r1-node-1": ("10.2.0.6",),
    }

    observation = GcloudTpuProvider(runner=_tpu_runner(uneven, "v6e-16")).ensure(_multislice_record("v6e-16"))

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.observed_tpu_vms == 3


def test_autoscale_prefers_the_catalog_over_the_suffix_derivation(monkeypatch) -> None:
    """The branch is invisible today, which is exactly why it needs a test.

    Current catalog entries agree with the suffix-derived answer, so an unusual
    host CPU count is needed to distinguish catalog lookup from fallback derivation.

    ``shape_for`` is patched, not the function under test: the branch's whole job
    is to consult the catalog, so substituting the catalog is what exercises it.
    """

    from eyvan.capacity.autoscale import profiles
    from eyvan.capacity.hardware_catalog import AcceleratorShape

    unusual = AcceleratorShape(
        accelerator_type="v5p-16",
        kind="tpu",
        generation="v5p",
        chips=8,
        quota_units=16,
        host_cpu=96,
        host_memory_gib=384,
        capacity_backend="gcp-tpu",
        multislice_capable=True,
        source_url="https://provider.example.com/tpu/v5p",
        source_version="2026-08-04",
    )
    monkeypatch.setattr(profiles, "shape_for", lambda _type: unusual)

    resolved = profiles.tpu_profile_from_accelerator_type("v5p-16")

    assert resolved is not None
    assert resolved.cpu == 96, "the suffix derivation's fixed 32 won over the catalog"
    assert resolved.memory_gib == 384
    assert resolved.tpu == 8


def test_an_uncatalogued_discovered_type_is_skipped_instead_of_getting_zero_memory(monkeypatch) -> None:
    """A quota suffix proves accelerator count and nothing about its TPU VM host."""

    from eyvan.capacity.autoscale import profiles

    monkeypatch.setattr(profiles, "shape_for", lambda _type: None)

    resolution = profiles.resolve_queued_tpu_profile("v5p-512")

    assert resolution.profile is None
    assert resolution.skip_reason is profiles.AutoscaleSkipReason.UNSUPPORTED_ACCELERATOR_TYPE


# --------------------------------------------------------------- multi-host slice rules


def test_a_generation_rule_answers_for_slice_sizes_no_row_names() -> None:
    """GCP lists 94 v5p types in one zone and this file named two of them.

    The other 92 were refused as types Eyvan "cannot derive a worker shape"
    for, which reads like a provider limitation and was a table-maintenance
    one. A multi-host slice is N of one VM, so the generation states that VM
    once.
    """

    derived = shape_for("v5p-512")

    assert derived is not None
    assert derived.chips == 256, "v5p bills two quota units per chip"
    assert derived.quota_units == 512
    assert (derived.host_cpu, derived.host_memory_gib) == (208, 448)
    assert derived.hosts == 64, "256 chips at four per ct5p-hightpu-4t"


def test_no_derived_number_is_new() -> None:
    """The rule widens where sourced numbers apply; it does not add any.

    If these ever disagree, one of the two was edited alone -- which is the
    three-sources-for-one-fact problem this catalog exists to end.
    """

    row = shape_for("v5p-32")
    derived = shape_for("v5p-1024")

    assert row is not None and derived is not None
    assert (derived.host_cpu, derived.host_memory_gib) == (row.host_cpu, row.host_memory_gib)
    assert derived.chips_per_host == row.chips_per_host


def test_an_explicit_row_still_wins_over_its_generation_rule() -> None:
    """v5p-8 is compiled as an integrated Compute Engine TPU VM. A rule that
    could overwrite that would silently undo a deliberate decision."""

    single = shape_for("v5p-8")

    assert single is not None
    assert single.capacity_backend == "gcp-compute"
    assert single.multislice_capable is False
    assert single.machine_type == "ct5p-hightpu-4t"


def test_a_generation_that_declines_to_state_hosts_derives_none() -> None:
    """The v6e-16 row omits `hosts` on purpose: Eyvan checks that number against
    the provider's endpoints, and no receipt here establishes v6e's. A rule that
    invented one would make Eyvan refuse a correctly provisioned slice."""

    derived = shape_for("v6e-64")

    assert derived is not None
    assert (derived.host_cpu, derived.host_memory_gib) == (180, 720)
    assert derived.hosts is None
    assert derived.chips_per_host is None


def test_a_slice_smaller_than_one_host_is_never_derived() -> None:
    """Below the floor a host is subdivided rather than replicated, and the
    resources of a fraction of a VM do not follow from the whole one."""

    assert shape_for("v6e-2") is None, "below the v6e rule's floor and named by no row"
    assert shape_for("v5p-12") is None, "six chips is not whole four-chip hosts"
    assert shape_for("v5litepod-2") is None, "below the v5e rule's floor and named by no row"


def test_a_rule_may_not_claim_a_slice_smaller_than_its_own_host() -> None:
    with pytest.raises(HardwareCatalogError, match="below one host"):
        GenerationRule(
            generation="v5p",
            kind="tpu",
            quota_units_per_chip=2,
            min_chips=2,
            chips_per_host=4,
            host_cpu=208,
            host_memory_gib=448,
            capacity_backend="gcp-tpu",
            source_url="https://cloud.google.com/tpu/docs/v5p",
            source_version="Last updated 2026-08-04 UTC",
        )


def test_every_shipped_rule_names_its_source() -> None:
    """Same bar the rows carry: an unsourced number is a guess with a comment."""

    rules = hardware_catalog().generation_rules

    assert rules, "the shipped catalog declares multi-host rules"
    for rule in rules:
        assert rule.source_url.startswith("https://")
        assert rule.source_version.strip()


def test_the_seventh_generation_is_named_tpu7x_not_v7() -> None:
    """Every generation before it is ``v<n>``; Google shipped this one as
    ``tpu7x``, and a parser anchored on a leading ``v`` refused the whole
    generation before any policy could look at it."""

    from eyvan.capacity.providers.gcp.tpu_types import parse_tpu_accelerator_type

    parsed = parse_tpu_accelerator_type("tpu7x-256")

    assert parsed is not None
    assert parsed.generation == "tpu7x"
    assert parsed.size == 256


def test_tpu7x_slices_resolve_from_the_published_vm(monkeypatch) -> None:
    """224 vCPU and 960 GB per VM, four chips, from Google's own page.

    The suffix counts chips here: v4 and v5p count TensorCores and are the
    exception, so reading 256 as 128 chips would ask for half the hardware.
    """

    shape = shape_for("tpu7x-256")

    assert shape is not None
    assert shape.chips == 256
    assert shape.quota_units == 256
    assert (shape.host_cpu, shape.host_memory_gib) == (224, 960)
    assert shape.hosts == 64, "the published table pairs 4x8x8 with 64 hosts"
    assert shape.machine_type == "tpu7x-standard-4t"


def test_a_tpu7x_slice_smaller_than_one_vm_is_refused() -> None:
    """ "All TPU7x slices use full-host, 4-chip VMs", so a smaller one is not a
    slice this repository can describe."""

    assert shape_for("tpu7x-2") is None


def test_the_autoscaler_will_generate_tpu7x_pools() -> None:
    """A shape the catalog knows is useless if the planner still refuses it."""

    from eyvan.capacity.autoscale.profiles import resolve_queued_tpu_profile

    resolution = resolve_queued_tpu_profile("tpu7x-64")

    assert resolution.skip_reason is None
    assert resolution.profile is not None
    assert resolution.profile.tpu == 64


def test_single_vm_v5e_and_v6e_shapes_are_sourced_rows_not_derivations() -> None:
    """Common Spot/TRC shapes: each is one published VM, so its numbers are exact.

    Before these rows every v5e type and v6e-1/-4 needed a hand-written
    ``resources:`` block, which is the guesswork this catalog exists to end.
    """

    expected = {
        "v6e-1": (1, 44, 176),
        "v6e-4": (4, 180, 720),
        "v5litepod-1": (1, 24, 48),
        "v5litepod-4": (4, 112, 192),
        "v5litepod-8": (8, 224, 384),
    }
    for accelerator_type, (chips, cpu, memory) in expected.items():
        shape = shape_for(accelerator_type)
        assert shape is not None, accelerator_type
        assert (shape.chips, shape.hosts, shape.chips_per_host) == (chips, 1, chips), accelerator_type
        assert (shape.host_cpu, shape.host_memory_gib) == (cpu, memory), accelerator_type
        assert shape.capacity_backend == "gcp-tpu"


def test_multi_host_v5e_derives_per_vm_resources_without_claiming_an_endpoint_count() -> None:
    """Google's v5e host column counts physical hosts, not four-chip VMs; claiming
    it would make Eyvan refuse a correctly provisioned slice."""

    for accelerator_type, chips in (("v5litepod-16", 16), ("v5litepod-256", 256)):
        shape = shape_for(accelerator_type)
        assert shape is not None
        assert shape.chips == shape.quota_units == chips
        assert (shape.host_cpu, shape.host_memory_gib) == (112, 192)
        assert shape.hosts is None
