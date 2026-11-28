# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Nothing that decides where work runs may know which cloud it runs on.

Eyvan is one cloud deep. The risk in that is not the driver -- a driver is meant
to know its provider -- it is everything *around* the driver quietly learning
GCP's vocabulary, so that the second cloud is a rewrite rather than a plugin.
The moment to catch that is before the second cloud exists, because afterwards
every leak is load-bearing.

These tests draw the line where the plan draws it: the pure scheduler, the
capacity planner, and the durable model may not name a provider to decide
anything. The drivers, the CLI's help text and the hardware catalogue may --
they are about a provider by construction.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "eyvan"

#: Files that decide *what happens*, as opposed to files that talk to one
#: provider. A provider name appearing in any of these is the leak.
DECIDING_MODULES = (
    "domain/scheduler.py",
    "domain/models.py",
    "domain/costing.py",
    "capacity/planning.py",
)

#: Provider vocabulary. Written as whole words so "compute" inside
#: "computed" and "aws" inside "flaws" do not read as leaks.
_PROVIDER_WORDS = re.compile(
    r"\b(gcp|gcloud|google|aws|ec2|azure|vmss|coreweave|lambda_labs)\b",
    re.IGNORECASE,
)


def _code_without_comments(path: pathlib.Path) -> str:
    """Return one module's executable text, with comments and docstrings removed.

    A provider named in a comment is documentation -- often the *reason* a
    neutral check exists, which is worth keeping. A provider named in code is a
    decision, and that is what this file is about.
    """

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            first = body[0].value
            if isinstance(first.value, str):
                docstrings.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    lines = []
    for number, line in enumerate(source.splitlines(), start=1):
        if number in docstrings:
            continue
        lines.append(line.split("#", 1)[0])
    return "\n".join(lines)


@pytest.mark.parametrize("module", DECIDING_MODULES)
def test_a_deciding_module_names_no_provider(module: str) -> None:
    """The scheduler must reach a second cloud without being edited.

    Every provider name here is a branch somebody has to find and generalise
    later, in the component where getting it wrong misplaces work rather than
    failing loudly.
    """

    path = PACKAGE / module
    assert path.exists(), f"{module} moved; update this boundary rather than deleting it"
    found = sorted({match.group(0).lower() for match in _PROVIDER_WORDS.finditer(_code_without_comments(path))})

    assert not found, f"{module} decides using provider names: {', '.join(found)}"


def test_the_durable_model_gates_multislice_on_capability_not_on_a_provider() -> None:
    """A second cloud that can allocate a gang must not be refused by a record.

    This was the one real leak: the model asked whether the provider was GCP,
    which reads as validation and behaves as a moat -- an AWS driver could
    implement capacity blocks perfectly and still be rejected by a validator in
    a file that has nothing to do with any cloud.
    """

    from eyvan.records import CapacityPoolRecord, ResourceVector

    resources = ResourceVector(cpu=8, memory_gib=32, tpu=8)
    pool = CapacityPoolRecord(
        id="pool-1",
        fleet_name="research-fleet",
        node_type="v6e-8",
        provider="a-cloud-nobody-has-written-yet",
        capacity_backend="gcp-tpu",
        provider_project="project",
        region="region-1",
        zone="region-1-a",
        capacity=resources,
        allocatable=resources,
        accelerator_type="v6e-8",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        min_cells=0,
        initial_cells=0,
        max_cells=4,
    )

    assert pool.hosts_multi_worker_cells, "the backend's capability decides, not the provider's name"


def test_a_backend_that_declares_nothing_hosts_one_machine_per_cell() -> None:
    """The safe default. A driver that has not said it can allocate a gang cannot."""

    from eyvan.capacity.drivers.base import backend_capabilities

    assert not backend_capabilities("a-new-cloud").multi_worker_cells
    assert backend_capabilities(None).max_workers_per_cell == 1


def test_the_multislice_ceiling_comes_from_the_backend_that_sets_it() -> None:
    """Each backend's ceiling is its own: 256 is GCP's queued-resource limit.

    The durable constant tracks the GCP TPU entry specifically, and the
    widest cell any backend builds comes from the table as a whole.
    """

    from eyvan.capacity.drivers.base import (
        _BACKEND_CAPABILITIES,
        backend_capabilities,
        max_multi_worker_cell_width,
    )
    from eyvan.records import MAX_GCP_TPU_MULTISLICE_WORKERS

    assert backend_capabilities("gcp-tpu").max_workers_per_cell == 256
    assert MAX_GCP_TPU_MULTISLICE_WORKERS == backend_capabilities("gcp-tpu").max_workers_per_cell
    assert max_multi_worker_cell_width() == max(
        capabilities.max_workers_per_cell for capabilities in _BACKEND_CAPABILITIES.values()
    )


def test_multi_worker_backends_declare_their_own_ceilings() -> None:
    """Kubernetes and the operator-hosted backends build wide cells too."""

    from eyvan.capacity.drivers.base import backend_capabilities

    for name in ("kubernetes", "onprem-ssh", "manual"):
        capabilities = backend_capabilities(name)
        assert capabilities.multi_worker_cells, name
        assert capabilities.max_workers_per_cell == 512, name
    for name in ("gcp-compute", "aws-ec2", "azure-vm"):
        assert not backend_capabilities(name).multi_worker_cells, name


def test_a_provider_becomes_eligible_by_declaring_a_capable_backend() -> None:
    """The property that makes this a plugin point rather than a list.

    Adding a cloud should mean adding a driver and one capability entry, not
    editing the durable model -- so the set of eligible providers is derived
    from the table and cannot drift from it.
    """

    from eyvan.capacity.drivers.base import (
        _BACKEND_CAPABILITIES,
        BackendCapabilities,
        capacity_backend_agent_family,
        providers_hosting_multi_worker_cells,
    )

    before = providers_hosting_multi_worker_cells()
    _BACKEND_CAPABILITIES["acme-gang"] = BackendCapabilities(multi_worker_cells=True, max_workers_per_cell=8)
    try:
        after = providers_hosting_multi_worker_cells()
    finally:
        del _BACKEND_CAPABILITIES["acme-gang"]

    assert after - before == {capacity_backend_agent_family("acme-gang")}
    assert providers_hosting_multi_worker_cells() == before, "the table is the only source"


def test_the_submission_path_gates_multislice_on_capability_too() -> None:
    """The same leak, one layer up, where it survived the durable model's cure.

    ``WorkloadDefinition`` was taught to ask the capability table, but the
    submission path in front of it still asked whether the pool was GCP -- so a
    second cloud that allocated a gang atomically would have satisfied the
    record and been refused by the thing that builds it. A moat is no less a
    moat for being one function earlier.
    """

    from eyvan.capacity.drivers.base import _BACKEND_CAPABILITIES, BackendCapabilities
    from eyvan.records import CapacityPoolRecord, ResourceLimits
    from eyvan.server.service.control_plane import _multislice_refusal

    def pool(**overrides) -> CapacityPoolRecord:
        fields = {
            "capacity": ResourceLimits(tpu=4),
            "allocatable": ResourceLimits(tpu=4),
            "executor": "eyvan-native",
            **overrides,
        }
        return CapacityPoolRecord(
            id="pool-1",
            fleet_name="research",
            node_type="gang-16",
            provider="acme",
            region="acme-east",
            agent_endpoint="http://agent.invalid",
            **fields,
        )

    assert _multislice_refusal(pool(capacity_backend="acme-gang")), "an undeclared backend allocates one machine"

    _BACKEND_CAPABILITIES["acme-gang"] = BackendCapabilities(multi_worker_cells=True, max_workers_per_cell=8)
    try:
        assert _multislice_refusal(pool(capacity_backend="acme-gang")) == "", (
            "a backend that declares atomic gang allocation is refused for being the wrong brand"
        )
        # The other two conditions are about the shape, not the cloud, and must
        # survive: ranks are placed across hosts by the native executor, and a
        # gang of machines with no accelerator in them is not worth allocating
        # atomically.
        assert _multislice_refusal(pool(capacity_backend="acme-gang", executor="container"))
        assert _multislice_refusal(
            pool(
                capacity_backend="acme-gang",
                capacity=ResourceLimits(cpu=8),
                allocatable=ResourceLimits(cpu=8),
            )
        )
    finally:
        del _BACKEND_CAPABILITIES["acme-gang"]


def test_a_pool_that_names_no_backend_keeps_the_answer_it_had() -> None:
    """Backward compatibility, and the reason the fallback exists.

    ``capacity_backend`` is optional and arrived after multislice did. Judging a
    pool that never set it as "one machine per cell" would take multislice away
    from every fleet applied before the field, on an upgrade, with no config
    change to point at.
    """

    from eyvan.capacity.drivers.base import providers_hosting_multi_worker_cells
    from eyvan.records import CapacityPoolRecord, ResourceLimits
    from eyvan.server.service.control_plane import _multislice_refusal

    provider = sorted(providers_hosting_multi_worker_cells())[0]
    legacy = CapacityPoolRecord(
        id="pool-legacy",
        fleet_name="research",
        node_type="v5p-8",
        provider=provider,
        # Required of this provider's pools by the record itself, and unrelated
        # to what is under test.
        provider_project="a-project",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceLimits(tpu=4),
        allocatable=ResourceLimits(tpu=4),
        executor="eyvan-native",
        agent_endpoint="http://agent.invalid",
    )

    assert _multislice_refusal(legacy) == ""
