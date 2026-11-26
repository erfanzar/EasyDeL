# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Focused tests for the live native v5p-16 acceptance harness helpers.

The harnesses live in the *monorepo's* ``scripts/`` directory, not in this
package -- they drive EasyDeL on real TPUs, so that is where they belong. But
``libs/eyvan`` is mirrored to a standalone repository, and in that checkout the
directory simply is not there.

So these skip with a stated reason rather than failing. A test that fails
because of where a checkout begins teaches contributors that a red suite is
normal, and that is a much more expensive habit than a skipped test.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from eyvan.capacity.drivers.base import CapacityStatus, CellObservation, MutationContext

#: The monorepo's script directory, which a standalone checkout of this package
#: does not have.
_MONOREPO_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _harness(name: str) -> Path:
    """Return one acceptance harness, or skip when this checkout has no monorepo."""

    script = _MONOREPO_SCRIPTS / name
    if not script.exists():
        pytest.skip(f"{name} lives in the monorepo's scripts/, which a standalone eyvan checkout does not include")
    return script


def _load_live_harness() -> ModuleType:
    """Load the standalone acceptance script without invoking its main entrypoint."""

    script = _harness("eyvan_native_v5p16_live.py")
    spec = importlib.util.spec_from_file_location("eyvan_native_v5p16_live_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _context() -> MutationContext:
    return MutationContext(idempotency_key="release-1", generation=2, fencing_token="cell-1-g2")


def test_release_capacity_retries_with_the_same_mutation_context(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _load_live_harness()
    contexts: list[MutationContext] = []

    class Driver:
        attempts = 0

        def release(self, cell_id: str, context: MutationContext) -> CellObservation:
            assert cell_id == "cell-1"
            self.attempts += 1
            contexts.append(context)
            if self.attempts == 1:
                raise RuntimeError("temporary delete failure")
            return CellObservation(cell_id=cell_id, state=CapacityStatus.RELEASED, generation=context.generation)

    sleeps: list[float] = []
    monkeypatch.setattr(harness.time, "sleep", sleeps.append)
    released = harness._release_capacity_with_retry(Driver(), "cell-1", _context(), timeout_s=30)

    assert released.state is CapacityStatus.RELEASED
    assert len(contexts) == 2
    assert contexts == [_context(), _context()]
    assert sleeps


def test_release_capacity_surfaces_an_exhausted_cleanup_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _load_live_harness()

    class Driver:
        def release(self, cell_id: str, context: MutationContext) -> None:
            assert cell_id == "cell-1"
            assert context == _context()
            raise RuntimeError("persistent delete failure")

    clock = iter((0.0, 0.0))
    monkeypatch.setattr(harness.time, "monotonic", lambda: next(clock))

    with pytest.raises(RuntimeError, match="did not finish within 0s") as exc_info:
        harness._release_capacity_with_retry(Driver(), "cell-1", _context(), timeout_s=0)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_live_harness_loads_native_runtime_from_the_same_fleet_yaml(tmp_path: Path) -> None:
    harness = _load_live_harness()
    fleet_path = tmp_path / "v5p16.yaml"
    fleet_path.write_text(
        textwrap.dedent(
            """\
            version: 1
            name: v5p-live
            provider:
              type: gcp
              project: project-a
              region: us-east5
              zone: us-east5-a
            agent_endpoint: http://agent.internal:8090
            executor: eyvan-native
            capacity_class: spot
            pools:
              v5p-16:
                label: v5p-16
                resources: {CPU: 32, TPU: 8}
                min_workers: 0
                max_workers: 0
                provider_options:
                  service_account: worker@project-a.iam.gserviceaccount.com
                  bootstrap_egress: private-google-access
                runtime:
                  env:
                    HF_HOME: /var/lib/eyvan/workload/hf
                  container:
                    image: registry.example.com/easydel:latest
                    privileged: true
                  artifacts:
                    - source: gs://runtime-artifacts/easydel.whl
                      destination: /opt/eyvan-runtime/easydel.whl
                  init_commands:
                    - /opt/eyvan-native/venv/bin/python -m pip install /opt/eyvan-runtime/easydel.whl
            """
        ),
        encoding="utf-8",
    )

    pool = harness.V5p16LivePool.from_fleet(fleet_path, "v5p-16")
    desired = pool.desired

    assert desired["project"] == "project-a"
    assert "node_count" not in desired["provider_options"]
    assert pool.runtime["env"]["HF_HOME"] == "/var/lib/eyvan/workload/hf"
    assert pool.runtime["artifacts"][0]["destination"] == "/opt/eyvan-runtime/easydel.whl"
    assert pool.runtime["container"]["image"] == "registry.example.com/easydel:latest"
    assert "startup_script" not in desired["provider_options"]
    assert pool.tpu_per_worker == 8


def test_live_harness_reports_one_worker_one_pod_without_slice_language() -> None:
    harness = _load_live_harness()
    result = harness.V5p16LivePool(
        name="v5p-16",
        accelerator_type="v5p-16",
        desired={},
        runtime={},
        tpu_per_worker=8,
    ).success_result()

    assert result == {
        "result": "passed",
        "pods": 1,
        "pool": "v5p-16",
        "type": "v5p-16",
        "workers": 1,
    }
    assert "slices" not in result


def test_oci_smoke_has_no_eyvan_runtime_import() -> None:
    """OCI images only need JAX and EasyDeL; bootstrap supplies multihost setup."""

    module = ast.parse(_harness("eyvan_easydel_tpu_smoke.py").read_text(encoding="utf-8"))
    imported_modules = {
        alias.name for node in ast.walk(module) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module or "" for node in ast.walk(module) if isinstance(node, ast.ImportFrom)}

    assert not any(name == "eyvan" or name.startswith("eyvan.") for name in imported_modules)


def test_no_test_depends_on_a_path_outside_this_package() -> None:
    """``libs/eyvan`` is mirrored to a standalone repository.

    A test that walks up past the package root passes in the monorepo and fails
    in the mirror, where the directory it reaches for does not exist. The two
    sites above were exactly that, and they failed rather than skipping -- so
    the mirror's suite was red for a reason no contributor there could act on.

    This is the guard, not the fix: reaching out is permitted when the reach is
    guarded by a skip, and forbidden when it is spelled inline where nothing can
    check it.
    """

    import re

    offenders: list[str] = []
    tests = Path(__file__).resolve().parent
    for path in sorted(tests.glob("*.py")):
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            # Comments are stripped before matching. A guard that reads prose
            # fires on its own explanation, which is how this one first failed:
            # the sentence describing the rule tripped the rule.
            line = raw.split("#", 1)[0]
            # A reach is guarded when it is bound to a named monorepo anchor,
            # because every such anchor here is paired with an existence check
            # that skips. An inline walk mid-expression is the unguarded shape.
            if re.search(r"parents\[[3-9]\]", line) and "_MONOREPO" not in line:
                offenders.append(f"{path.name}:{number}")

    assert not offenders, (
        "these tests reach outside libs/eyvan without a guarded skip, so they fail in the standalone "
        "mirror rather than skipping: " + ", ".join(offenders)
    )


def test_the_harness_lookup_skips_rather_than_failing_when_the_monorepo_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mirror's suite must be green, not red-for-a-structural-reason.

    Asserted by running the lookup against a directory that has no harnesses,
    which is what a standalone checkout looks like -- rather than by trusting
    that the ``exists()`` branch does what it reads like.
    """

    monkeypatch.setattr(sys.modules[__name__], "_MONOREPO_SCRIPTS", tmp_path)

    with pytest.raises(BaseException) as raised:
        _harness("eyvan_native_v5p16_live.py")

    assert raised.typename == "Skipped"
    assert "standalone eyvan checkout" in str(raised.value)
