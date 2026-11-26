# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One place that knows what an accelerator shape is.

Shape knowledge lived in three places that could disagree. ``fleet_config`` held
a compute-TPU dict and a queued-slice dict; ``autoscale.profiles`` derived a chip
count from the type suffix and then hardcoded ``cpu=32`` for every TPU ever made,
which its own docstring admitted was "the same conservative CPU vector until
host-specific discovery is available"; and every fleet YAML restated the answer
by hand in ``resources:``.

Three sources for one fact is not redundancy, it is three chances to be wrong in
different directions -- and the operator writing the fourth copy into YAML has no
way to know which of the three they are supposed to agree with.

**Not generated from quota discovery.** ``quota_discovery`` reads quota
*metrics*, which give a generation and a unit ceiling and never a host count.
Quota discovery stays the authority for what is *available*; this is the
authority for what a type *is*. They join on the accelerator type.

**Where the catalog stops.** ``hosts`` and ``chips_per_host`` are optional and
recorded only where this repository grounds them. Making them mandatory would
mean inventing topology for generations nothing here describes, and a wrong host
count makes Eyvan reject a slice the provider built correctly -- a worse failure
than not checking, because it looks like a provider fault.

CPU and memory are different: both participate in admission, so every built-in
shape must carry positive per-VM values plus the provider document and version
that establish them. A GPU entry has the same requirement. Hardware the catalog
does not describe remains usable only through an explicit, complete fleet
resource vector; provider discovery alone is not evidence for host resources.

**GPU cell profiles are keyed by machine type, not by GPU model.** A cell is a
host, and on GCP the GPU *model* (``nvidia-l4``) fixes none of the three numbers
admission needs: ``g2-standard-4`` and ``g2-standard-96`` carry the same L4 and
differ by 24x in vCPU and 8x in attached devices. The machine type fixes all
three, so it is the key, and ``gpu_model`` records which accelerator is attached.
This also means no name that an existing pool already puts in
``accelerator_type`` gains a catalog entry here, so no deployment's write
boundary changes meaning on upgrade.

**Nothing in this file is hardware-qualified.** The GPU rows are transcribed from
the provider's published machine-shape table. No GPU has executed an Eyvan
workload, so treat a GPU row as a reviewable description of a host, not as
evidence that a run on that host works.
"""

from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml

#: Schema version this build understands. A catalog declaring anything else is
#: refused rather than partially read: a build silently ignoring fields it does
#: not know would size pools from half a description.
SUPPORTED_CATALOG_VERSION = 3

CATALOG_PATH = Path(__file__).resolve().parent / "data" / "accelerator_shapes.yaml"


class HardwareCatalogError(ValueError):
    """Raised when the catalog cannot be loaded or does not describe itself."""


#: Accelerator kinds this catalog can describe. Anything else is refused rather
#: than stored: ``kind`` selects the resource dimension a shape is counted in
#: (``schedulable_tpu`` / ``schedulable_gpu``) and the admission boundary reads
#: it with ``getattr``, so an unrecognised kind would silently count as nothing.
SUPPORTED_KINDS: tp.Final = frozenset({"gpu", "tpu"})

#: Capacity backends that provision an ordinary VM. A GPU host is one of these:
#: there is no queued-resource path for GPUs, and a GPU row declaring the Cloud
#: TPU backend would compile into a request no driver can serve.
_COMPUTE_BACKENDS: tp.Final = frozenset({"compute", "gcp", "gcp-compute"})


@dataclass(frozen=True, slots=True)
class AcceleratorShape:
    """The dimensions of one accelerator type.

    ``chips`` is what a workload gets; ``quota_units`` is what the provider
    counts. They differ by generation -- v4 and v5p bill two units per chip,
    v5e/v6e one -- and conflating them is why a v5p-16 pool asking for 16 TPUs
    would silently request twice the hardware.

    ``host_cpu`` and ``host_memory_gib`` are the resources of one VM host, not a
    sum across a multi-host slice. Eyvan launches the native workload on every
    host, so admission must ensure the per-host request fits each host.
    ``source_url`` and ``source_version`` make those otherwise easy-to-guess
    numbers reviewable and make unsourced future accelerator entries invalid.

    A ``kind="gpu"`` row obeys every rule a TPU row obeys and three more, because
    a GPU host is one VM rather than a slice: it must record ``hosts=1`` with
    ``chips_per_host`` equal to ``chips`` (so per-rank device assignment has an
    exact device count instead of a guess), it must name the attached
    ``gpu_model``, and it must name the ``machine_type`` that fixes its CPU,
    memory and device count. Refusing at construction rather than defaulting is
    the same rule TPU host resources already obey: a shape that will not state
    its host is not a shape, and a zero would make every workload unschedulable
    on a pool that looks healthy.

    No GPU row is hardware-qualified. Nothing has executed on a GPU under Eyvan.
    """

    accelerator_type: str
    kind: str
    generation: str
    chips: int
    quota_units: int
    host_cpu: int
    host_memory_gib: float
    capacity_backend: str
    multislice_capable: bool
    source_url: str
    source_version: str
    hosts: int | None = None
    chips_per_host: int | None = None
    machine_type: str | None = None
    tpu_type: str | None = None
    image_family: str | None = None
    image_project: str | None = None
    gpu_model: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_KINDS:
            known = ", ".join(sorted(SUPPORTED_KINDS))
            raise HardwareCatalogError(f"{self.accelerator_type}: kind {self.kind!r} is not one of {known}")
        if self.chips <= 0 or self.quota_units <= 0:
            raise HardwareCatalogError(f"{self.accelerator_type}: chips and quota_units must be positive")
        # A whole number, because a host has whole CPUs and ``TpuProfile``
        # requires a positive integer. Storing a float here would put the
        # coercion at every call site instead of at the one boundary.
        if isinstance(self.host_cpu, bool) or not isinstance(self.host_cpu, int) or self.host_cpu <= 0:
            raise HardwareCatalogError(f"{self.accelerator_type}: host_cpu must be a positive integer")
        if (
            isinstance(self.host_memory_gib, bool)
            or not isinstance(self.host_memory_gib, int | float)
            or not math.isfinite(self.host_memory_gib)
            or self.host_memory_gib <= 0
        ):
            raise HardwareCatalogError(f"{self.accelerator_type}: host_memory_gib must be a positive finite number")
        source = urlparse(self.source_url) if isinstance(self.source_url, str) else None
        if source is None or source.scheme != "https" or not source.netloc:
            raise HardwareCatalogError(f"{self.accelerator_type}: source_url must be an absolute HTTPS URL")
        if (
            not isinstance(self.source_version, str)
            or not self.source_version.strip()
            or self.source_version != self.source_version.strip()
        ):
            raise HardwareCatalogError(f"{self.accelerator_type}: source_version must be a non-empty trimmed string")
        if (self.hosts is None) != (self.chips_per_host is None):
            raise HardwareCatalogError(
                f"{self.accelerator_type}: hosts and chips_per_host are recorded together or not at all"
            )
        if self.hosts is not None and self.chips_per_host is not None:
            # The invariant that turns a typo into a CI failure rather than a
            # pool that provisions the wrong number of machines.
            if self.chips != self.hosts * self.chips_per_host:
                raise HardwareCatalogError(
                    f"{self.accelerator_type}: {self.chips} chips is not {self.hosts} hosts "
                    f"x {self.chips_per_host} chips per host"
                )
        if self.kind == "gpu":
            self._validate_gpu_host()
        if self.kind == "tpu":
            if self.gpu_model is not None:
                raise HardwareCatalogError(f"{self.accelerator_type}: a TPU shape cannot name a gpu_model")
            # Imported here, not at module scope. ``tpu_types`` lives under the
            # GCP provider package, whose ``__init__`` reaches back into modules
            # that consume this catalog -- so a module-level import makes the
            # cycle real and its breakage depend on which module a process
            # happens to import first. The catalog stays a leaf.
            from eyvan.capacity.providers.gcp.tpu_types import parse_tpu_accelerator_type

            parsed = parse_tpu_accelerator_type(self.accelerator_type)
            if parsed is None:
                raise HardwareCatalogError(f"{self.accelerator_type}: not a valid Cloud TPU accelerator type")
            # The suffix *is* the quota unit count. Recording a different number
            # would make the catalog disagree with the provider's own naming.
            if parsed.size != self.quota_units:
                raise HardwareCatalogError(
                    f"{self.accelerator_type}: declares {self.quota_units} quota units but its name says {parsed.size}"
                )
            if parsed.generation != self.generation:
                raise HardwareCatalogError(
                    f"{self.accelerator_type}: declares generation {self.generation!r} "
                    f"but its name says {parsed.generation!r}"
                )

    def _validate_gpu_host(self) -> None:
        """Refuse a GPU row that does not describe one addressable VM host.

        Every clause is here because the value it guards is read later by
        something that cannot ask again. ``chips_per_host`` becomes a rank's
        device list; ``machine_type`` is what a provider is asked to create;
        ``gpu_model`` is the only record of which accelerator a pool holds once
        the fleet document is compiled away.

        Raises:
            HardwareCatalogError: If the row is not a complete single-VM GPU
                host description.
        """

        if self.hosts is None or self.chips_per_host is None:
            raise HardwareCatalogError(
                f"{self.accelerator_type}: a GPU shape must record hosts and chips_per_host; "
                "a rank cannot be given an exact device list from an unknown device count"
            )
        if self.hosts != 1:
            raise HardwareCatalogError(f"{self.accelerator_type}: a GPU shape describes one VM host, not {self.hosts}")
        # GPU quota is counted one unit per attached device, unlike v4/v5p TPU
        # quota. Recording anything else would size a pool against a ceiling the
        # provider never applies.
        if self.quota_units != self.chips:
            raise HardwareCatalogError(
                f"{self.accelerator_type}: a GPU shape bills one quota unit per device, "
                f"so {self.quota_units} quota units cannot describe {self.chips} devices"
            )
        if self.multislice_capable:
            raise HardwareCatalogError(f"{self.accelerator_type}: multislice is a Cloud TPU concept, not a GPU one")
        backend = self.capacity_backend.strip().lower().replace("_", "-")
        if backend not in _COMPUTE_BACKENDS:
            raise HardwareCatalogError(
                f"{self.accelerator_type}: a GPU shape is provisioned as a VM, "
                f"so capacity_backend {self.capacity_backend!r} cannot serve it"
            )
        for field_name in ("gpu_model", "machine_type"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise HardwareCatalogError(
                    f"{self.accelerator_type}: a GPU shape must name a non-empty trimmed {field_name}"
                )
        if self.tpu_type is not None:
            raise HardwareCatalogError(f"{self.accelerator_type}: a GPU shape cannot name a tpu_type")

    @property
    def schedulable_tpu(self) -> int:
        """Return the TPU resource count Eyvan schedules against.

        Chips, not quota units. A workload asks for TPUs it can compute on.
        """

        return self.chips if self.kind == "tpu" else 0

    @property
    def schedulable_gpu(self) -> int:
        """Return the GPU resource count Eyvan schedules against.

        Zero for a TPU shape, for the same reason :attr:`schedulable_tpu` is
        zero for a GPU one: a caller asking "how much of dimension X does this
        shape offer" must get an answer that is safe to add up.
        """

        return self.chips if self.kind == "gpu" else 0

    @property
    def devices_per_host(self) -> int | None:
        """Return accelerators on one host, or ``None`` when it is not recorded.

        ``None`` is a real answer and must stay one. A GPU row always records it
        because per-rank device assignment cannot be derived without it; several
        TPU rows deliberately do not, and turning that absence into a number is
        how Eyvan would start pinning ranks to devices that do not exist.
        """

        return self.chips_per_host


@dataclass(frozen=True, slots=True)
class GenerationRule:
    """How one TPU generation builds a multi-host slice out of one VM.

    A multi-host slice is not new hardware, it is ``N`` of one machine type
    wired together -- so the per-VM resources admission needs are a property of
    the generation, not of each size. Stating them once is what lets Eyvan
    accept a slice size Google added yesterday instead of refusing it until
    somebody transcribes a row.

    ``min_chips`` is never 1. Below it a host is *subdivided* rather than
    replicated, and the resources of a fraction of a VM do not follow from the
    whole one; those sizes stay explicit rows or stay unknown. Deriving them
    would be inventing numbers.

    Attributes:
        quota_units_per_chip: What the provider bills per chip -- two for v4
            and v5p, one for v5e and v6e. The accelerator suffix counts these,
            not chips, and conflating them requests twice the hardware.
        chips_per_host: Chips on one VM of this generation's multi-host shape.
        establishes_hosts: Whether dividing by ``chips_per_host`` yields the
            endpoint count Eyvan may *check*, and not merely the number of VMs
            the resources describe. Off by default, and it must stay off for a
            generation whose explicit rows decline to state ``hosts``: Eyvan
            compares that number against the provider's endpoints, so a derived
            one that is wrong makes it refuse a slice that was built correctly
            -- the failure the explicit rows avoid by staying silent.
    """

    generation: str
    kind: str
    quota_units_per_chip: int
    min_chips: int
    chips_per_host: int
    host_cpu: int
    host_memory_gib: float
    capacity_backend: str
    source_url: str
    source_version: str
    machine_type: str | None = None
    establishes_hosts: bool = False

    def __post_init__(self) -> None:
        if self.kind != "tpu":
            raise HardwareCatalogError(f"{self.generation}: generation rules describe TPU slices only")
        for name in ("quota_units_per_chip", "min_chips", "chips_per_host", "host_cpu"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise HardwareCatalogError(f"{self.generation}: {name} must be a positive integer")
        if not isinstance(self.host_memory_gib, (int, float)) or self.host_memory_gib <= 0:
            raise HardwareCatalogError(f"{self.generation}: host_memory_gib must be positive")
        if self.min_chips < self.chips_per_host:
            # A slice smaller than one VM is a subdivided host, which is exactly
            # what this rule may not describe.
            raise HardwareCatalogError(
                f"{self.generation}: min_chips {self.min_chips} is below one host's {self.chips_per_host} chips"
            )
        if self.min_chips % self.chips_per_host:
            raise HardwareCatalogError(f"{self.generation}: min_chips must be whole hosts")
        for name in ("source_url", "source_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise HardwareCatalogError(f"{self.generation}: a generation rule must name a {name}")

    def derive(self, accelerator_type: str, quota_units: int) -> AcceleratorShape | None:
        """Return the shape for one slice of this generation, if the rule covers it.

        ``None`` rather than a guess whenever the size is not whole VMs of this
        generation: a slice Eyvan cannot divide evenly is one whose hosts it
        would be describing wrongly, and a wrong host count makes Eyvan refuse
        capacity that was provisioned correctly.
        """

        if quota_units % self.quota_units_per_chip:
            return None
        chips = quota_units // self.quota_units_per_chip
        if chips < self.min_chips or chips % self.chips_per_host:
            return None
        return AcceleratorShape(
            accelerator_type=accelerator_type,
            kind=self.kind,
            generation=self.generation,
            chips=chips,
            quota_units=quota_units,
            hosts=chips // self.chips_per_host if self.establishes_hosts else None,
            chips_per_host=self.chips_per_host if self.establishes_hosts else None,
            host_cpu=self.host_cpu,
            host_memory_gib=self.host_memory_gib,
            capacity_backend=self.capacity_backend,
            multislice_capable=True,
            machine_type=self.machine_type,
            source_url=self.source_url,
            source_version=self.source_version,
        )


@dataclass(frozen=True, slots=True)
class HardwareCatalog:
    """Every shape this build knows, and the revision they came from."""

    catalog_version: int
    revision: str
    shapes: tuple[AcceleratorShape, ...]
    generation_rules: tuple[GenerationRule, ...] = ()

    def get(self, accelerator_type: str) -> AcceleratorShape | None:
        """Return one shape, or ``None`` for a type this catalog does not describe.

        An explicit row wins over a generation rule, always. The row carries
        what a rule cannot -- image families, machine types, and the deliberate
        exclusions -- and a rule that could override one would silently undo a
        decision somebody made on purpose.
        """

        normalized = accelerator_type.strip().lower()
        explicit = next((shape for shape in self.shapes if shape.accelerator_type == normalized), None)
        if explicit is not None:
            return explicit
        return self._derived(normalized)

    def _derived(self, normalized: str) -> AcceleratorShape | None:
        """Build a shape from this generation's multi-host rule, if one covers it."""

        from eyvan.capacity.providers.gcp.tpu_types import parse_tpu_accelerator_type

        parsed = parse_tpu_accelerator_type(normalized)
        if parsed is None:
            return None
        rule = next((item for item in self.generation_rules if item.generation == parsed.generation), None)
        return None if rule is None else rule.derive(normalized, parsed.size)

    def require(self, accelerator_type: str) -> AcceleratorShape:
        """Return one shape or refuse, naming what the catalog does know.

        Listing the alternatives is the difference between an error an operator
        can act on and one they have to go read source for.
        """

        shape = self.get(accelerator_type)
        if shape is not None:
            return shape
        known = ", ".join(sorted(item.accelerator_type for item in self.shapes))
        raise HardwareCatalogError(
            f"accelerator type {accelerator_type!r} is not in hardware catalog revision {self.revision}; "
            f"known types: {known}"
        )

    @property
    def accelerator_types(self) -> tuple[str, ...]:
        return tuple(shape.accelerator_type for shape in self.shapes)


def _load(path: Path) -> HardwareCatalog:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HardwareCatalogError(f"could not read the hardware catalog at {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise HardwareCatalogError(f"hardware catalog at {path} is not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise HardwareCatalogError(f"hardware catalog at {path} must be a mapping")

    version = document.get("catalog_version")
    if version != SUPPORTED_CATALOG_VERSION:
        raise HardwareCatalogError(
            f"hardware catalog declares schema version {version!r}; this build understands {SUPPORTED_CATALOG_VERSION}"
        )
    revision = document.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise HardwareCatalogError("hardware catalog must declare a non-empty revision")

    entries = document.get("shapes")
    if not isinstance(entries, list) or not entries:
        raise HardwareCatalogError("hardware catalog must declare at least one shape")

    shapes: list[AcceleratorShape] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise HardwareCatalogError("every hardware catalog shape must be a mapping")
        try:
            shape = AcceleratorShape(**entry)
        except TypeError as exc:
            raise HardwareCatalogError(f"hardware catalog shape {entry.get('accelerator_type')!r}: {exc}") from exc
        if shape.accelerator_type in seen:
            # Two entries for one type is the failure this catalog exists to
            # end, so it must not be reintroducible inside the catalog itself.
            raise HardwareCatalogError(f"hardware catalog declares {shape.accelerator_type!r} more than once")
        seen.add(shape.accelerator_type)
        shapes.append(shape)

    rules: list[GenerationRule] = []
    declared = document.get("generation_rules", [])
    if not isinstance(declared, list):
        raise HardwareCatalogError("hardware catalog generation_rules must be a list")
    for entry in declared:
        if not isinstance(entry, dict):
            raise HardwareCatalogError("every hardware catalog generation rule must be a mapping")
        try:
            rule = GenerationRule(**entry)
        except TypeError as exc:
            raise HardwareCatalogError(f"hardware catalog rule {entry.get('generation')!r}: {exc}") from exc
        if any(item.generation == rule.generation for item in rules):
            raise HardwareCatalogError(f"hardware catalog declares a rule for {rule.generation!r} more than once")
        rules.append(rule)

    return HardwareCatalog(
        catalog_version=version,
        revision=revision.strip(),
        shapes=tuple(shapes),
        generation_rules=tuple(rules),
    )


@lru_cache(maxsize=1)
def hardware_catalog() -> HardwareCatalog:
    """Return the catalog shipped with this build.

    Cached because it is read on every fleet parse and pool upsert, and it is
    immutable data inside the wheel.
    """

    return _load(CATALOG_PATH)


def load_hardware_catalog(path: str | Path) -> HardwareCatalog:
    """Load a catalog from an explicit path, for tests and for validating edits."""

    return _load(Path(path))


def shape_for(accelerator_type: str) -> AcceleratorShape | None:
    """Return the shipped catalog's entry for one accelerator type, if it has one."""

    return hardware_catalog().get(accelerator_type)


class _AdmissionResourceVector(tp.Protocol):
    """The built-in resource dimensions needed at an accelerator write boundary."""

    cpu: float
    memory_gib: float
    gpu: int
    tpu: int


def _advertised_accelerator_kinds(resources: _AdmissionResourceVector) -> set[str]:
    return {kind for kind in ("gpu", "tpu") if getattr(resources, kind) > 0}


def _gcp_accelerator_kind(accelerator_type: str, shape: AcceleratorShape | None) -> str | None:
    """Classify provider names whose kind is independently knowable."""

    if shape is not None:
        return shape.kind
    from eyvan.capacity.providers.gcp.tpu_types import parse_tpu_accelerator_type

    if accelerator_type == "tpu" or parse_tpu_accelerator_type(accelerator_type) is not None:
        return "tpu"
    if accelerator_type == "gpu" or accelerator_type.startswith("nvidia-"):
        return "gpu"
    return None


def validate_accelerator_write_boundary(
    *,
    provider: str,
    capacity_backend: str | None,
    accelerator_type: str | None,
    capacity: _AdmissionResourceVector,
    allocatable: _AdmissionResourceVector,
    boundary: str,
) -> None:
    """Validate accelerator identity and admission resources before persistence.

    Durable records deliberately do not call this function: historical rows with
    incomplete vectors must remain readable so ``eyvan doctor`` can name and
    repair them. Public cell/pool requests and the typed fleet compiler do call
    it, so no new write can bypass catalog kind/count semantics.

    Known GCP shapes fix only accelerator kind and count. Positive CPU and memory
    remain documented admission overrides. An uncatalogued GCP type stays usable
    through the documented manual escape hatch, but both capacity and allocatable
    must then be complete positive host vectors.
    """

    normalized_provider = provider.strip().lower()
    normalized_backend = (capacity_backend or "").strip().lower().replace("_", "-")
    normalized_type = accelerator_type.strip().lower() if accelerator_type is not None else None
    gcp_backends = {"gcp", "gcp-tpu", "gcp-attach", "tpu", "gcp-compute", "compute"}
    if normalized_backend in gcp_backends and normalized_provider != "gcp":
        raise ValueError(f"{boundary} capacity_backend {capacity_backend!r} requires provider 'gcp', not {provider!r}")
    type_declares_accelerator = normalized_type not in {None, "cpu"}
    backend_declares_tpu = normalized_backend in {"gcp-tpu", "gcp-attach", "tpu"}
    vector_declares_accelerator = any(
        amount > 0 for resources in (capacity, allocatable) for amount in (resources.gpu, resources.tpu)
    )
    if not (type_declares_accelerator or backend_declares_tpu or vector_declares_accelerator):
        return
    if not type_declares_accelerator or normalized_type is None:
        raise ValueError(f"{boundary} accelerator resources require a non-CPU accelerator_type")

    shape = shape_for(normalized_type) if normalized_provider == "gcp" else None
    capacity_kinds = _advertised_accelerator_kinds(capacity)
    allocatable_kinds = _advertised_accelerator_kinds(allocatable)
    vectors_are_complete = (
        capacity.cpu > 0
        and capacity.memory_gib > 0
        and allocatable.cpu > 0
        and allocatable.memory_gib > 0
        and len(capacity_kinds) == 1
        and allocatable_kinds == capacity_kinds
    )
    if normalized_provider == "gcp" and shape is None and not vectors_are_complete:
        raise ValueError(
            f"{boundary} unknown GCP accelerator shape {accelerator_type!r} requires complete manual capacity and "
            "allocatable vectors with positive cpu, memory_gib, and the same one of gpu or tpu"
        )

    for field_name, resources in (("capacity", capacity), ("allocatable", allocatable)):
        if resources.cpu <= 0 or resources.memory_gib <= 0:
            raise ValueError(f"{boundary} {field_name} requires positive cpu and memory_gib for accelerator admission")
    if len(capacity_kinds) != 1 or allocatable_kinds != capacity_kinds:
        raise ValueError(f"{boundary} capacity and allocatable must advertise the same one of gpu or tpu")
    kind = next(iter(capacity_kinds))

    if normalized_provider == "gcp":
        expected_kind = _gcp_accelerator_kind(normalized_type, shape)
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(
                f"{boundary} accelerator_type {accelerator_type!r} is a {expected_kind} type, "
                f"but capacity and allocatable advertise {kind}"
            )
        if normalized_backend in {"gcp-tpu", "gcp-attach", "tpu"} and kind != "tpu":
            raise ValueError(f"{boundary} capacity_backend {capacity_backend!r} requires tpu resources, not {kind}")
        if normalized_backend in {"gcp-compute", "compute"} and kind == "tpu":
            # Compute-capable when the catalog names a single VM's machine type and
            # its Compute TPU identity, whatever the shape's default backend is: v5e
            # defaults to the queue, and a project the queue refuses can still run
            # it as Compute Spot VMs of the same machine family.
            integrated_tpu = shape is not None and (
                shape.capacity_backend in {"gcp-compute", "compute"}
                or (shape.machine_type is not None and shape.tpu_type is not None and shape.hosts == 1)
            )
            if not integrated_tpu:
                raise ValueError(
                    f"{boundary} capacity_backend {capacity_backend!r} cannot provision TPU type "
                    f"{accelerator_type!r}; use gcp-tpu/gcp or a cataloged gcp-compute TPU shape"
                )
        if shape is not None:
            expected_count = shape.chips
            for field_name, resources in (("capacity", capacity), ("allocatable", allocatable)):
                if getattr(resources, shape.kind) != expected_count:
                    raise ValueError(
                        f"{boundary} {field_name}.{shape.kind} must equal catalog count {expected_count} "
                        f"for {shape.accelerator_type!r}"
                    )


__all__: tp.Final = (
    "CATALOG_PATH",
    "SUPPORTED_CATALOG_VERSION",
    "SUPPORTED_KINDS",
    "AcceleratorShape",
    "HardwareCatalog",
    "HardwareCatalogError",
    "hardware_catalog",
    "load_hardware_catalog",
    "shape_for",
    "validate_accelerator_write_boundary",
)
