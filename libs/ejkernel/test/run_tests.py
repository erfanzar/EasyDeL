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


"""Test runner for ejkernels test suite.

Usage:
    python test/run_tests.py
    python test/run_tests.py --all
    python test/run_tests.py --modules
    python test/run_tests.py --operations
    python test/run_tests.py --kernels
    python test/run_tests.py --xla
    python test/run_tests.py --pallas
    python test/run_tests.py --triton
    python test/run_tests.py --cuda
    python test/run_tests.py --types
    python test/run_tests.py --verbose
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

# When running `python test/run_tests.py`, Python puts `test/` at the front of
# `sys.path`, which can shadow stdlib modules (e.g. `test/types` -> `types`).
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _THIS_DIR]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse  # noqa
from pathlib import Path  # noqa

import pytest  # noqa


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    selected: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists() or resolved in seen:
            continue
        seen.add(resolved)
        selected.append(path)
    return selected


def main():
    """Run test suite with specified options."""
    parser = argparse.ArgumentParser(description="Run ejkernels test suite")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument(
        "--modules",
        action="store_true",
        help="Run module-level tests (test/*.py + test/modules)",
    )
    parser.add_argument(
        "--operations",
        action="store_true",
        help="Run module operation tests (test/modules/operations)",
    )
    parser.add_argument("--kernels", action="store_true", help="Run all kernel tests (test/kernels)")
    parser.add_argument("--xla", action="store_true", help="Run only XLA kernel tests")
    parser.add_argument("--pallas", action="store_true", help="Run only Pallas kernel tests")
    parser.add_argument("--triton", action="store_true", help="Run only Triton kernel tests")
    parser.add_argument("--cuda", action="store_true", help="Run only CUDA kernel tests")
    parser.add_argument("--types", action="store_true", help="Run only types tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-k", "--keyword", type=str, help="Only run tests matching keyword")
    parser.add_argument("--failfast", action="store_true", help="Stop on first failure")
    parser.add_argument("--tb", type=str, default="short", help="Traceback style (short/long/native)")

    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent
    repo_root = test_dir.parent

    pytest_args: list[str] = []

    any_suite_selected = any(
        [
            args.all,
            args.modules,
            args.operations,
            args.kernels,
            args.xla,
            args.pallas,
            args.triton,
            args.cuda,
            args.types,
        ]
    )

    paths: list[Path] = []
    if args.all or not any_suite_selected:
        paths = [test_dir]
    else:
        if args.modules:
            paths.append(test_dir / "modules")
            paths.extend(sorted(test_dir.glob("test_*.py")))

        if args.operations and not args.modules:
            paths.append(test_dir / "modules" / "operations")

        kernel_paths: list[Path] = []
        if args.xla:
            kernel_paths.append(test_dir / "kernels" / "_xla")
        if args.pallas:
            kernel_paths.append(test_dir / "kernels" / "_pallas")
        if args.triton:
            kernel_paths.append(test_dir / "kernels" / "_triton")
        if args.cuda:
            kernel_paths.append(test_dir / "kernels" / "_cuda")
        if args.types:
            kernel_paths.append(test_dir / "kernels" / "types")
        if args.kernels and not kernel_paths:
            kernel_paths.append(test_dir / "kernels")

        paths.extend(kernel_paths)

    paths = _unique_existing_paths(paths)
    if not paths:
        raise SystemExit("No test paths selected (did you move the test suite?)")

    for path in paths:
        pytest_args.append(str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path))

    if args.verbose:
        pytest_args.append("-v")

    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    if args.failfast:
        pytest_args.append("-x")

    pytest_args.append(f"--tb={args.tb}")

    pytest_args.append("-ra")

    print("=" * 70)
    print("Running ejkernels test suite")
    print("=" * 70)
    print(f"Test paths: {' '.join(pytest_args[: len(paths)])}")
    print(f"Options: {' '.join(pytest_args[len(paths) :])}")
    print("=" * 70)
    print()

    exit_code = pytest.main(pytest_args)

    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
