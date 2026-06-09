# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Benchmark registry coverage tests."""

from pathlib import Path

import ejkernel.kernels  # noqa: F401
from benchmarks._op_benchmark_registry import SPECS
from ejkernel.kernels._registry import kernel_registry


def test_every_registered_algorithm_has_benchmark_spec():
    """Every registered kernel algorithm has a benchmark specification."""

    registered = set(kernel_registry.list_algorithms())
    covered = {spec.algorithm for spec in SPECS.values()}
    assert sorted(registered - covered) == []


def test_every_benchmark_spec_has_script_entrypoint():
    """Every benchmark specification has a compatibility script entry point."""

    root = Path(__file__).resolve().parents[1]
    missing = []
    for name in SPECS:
        path = root / "benchmarks" / f"benchmark_{name}.py"
        if not path.exists():
            missing.append(name)
    assert missing == []


def test_no_duplicate_benchmark_entrypoints():
    """Only canonical benchmark entry points and intentional standalone tools are kept."""

    root = Path(__file__).resolve().parents[1]
    allowed = {f"benchmark_{name}.py" for name in SPECS}
    allowed.update(
        {
            "_op_benchmark_registry.py",
            "benchmark_quantized_matmul_native_vs_gemlite.py",
            "benchmark_suite.py",
            "plot_qmm_native_vs_gemlite.py",
        }
    )
    actual = {path.name for path in (root / "benchmarks").glob("*.py")}
    assert sorted(actual - allowed) == []
