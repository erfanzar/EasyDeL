# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
"""Format EasyDeL Python code with ruff and (re)generate the Sphinx API docs.

Drives three steps the project runs from pre-commit and CI:

1. ``format_code`` — runs ``ruff check --fix`` (when ``--fix``) and
   ``ruff format`` over every ``*.py`` under ``--directory``.
2. ``generate_api_docs`` — walks the package, emits one ``.rst`` page per
   module mirroring the package layout under ``docs/api_docs/``, and writes
   per-package and root index pages with sorted toctree entries.
3. ``run_tests`` — runs ``pytest`` over ``test/`` when ``--test`` is set.

By default (no flags), every step except tests is executed.

Side effects:
    - Mutates files in place under ``--directory`` via ruff fixes.
    - Removes and recreates ``docs/api_docs/`` (when ``--clean``).

Usage:
    python scripts/format_and_generate_docs.py            # format + docs
    python scripts/format_and_generate_docs.py --all      # format + docs + tests
    python scripts/format_and_generate_docs.py --no-fix --docs
"""

import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from eformer.aparser import DataClassArgumentParser

PROJECT_NAME = "easydel"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_API_DIR = PROJECT_ROOT / "docs" / "api_docs"


def _strip_prefix(module_path: str) -> str:
    """Strip the top-level ``"<PROJECT_NAME>."`` prefix from a dotted module path.

    Args:
        module_path: Dotted Python module path.

    Returns:
        str: ``module_path`` with the leading ``"easydel."`` removed, unchanged
            when the prefix is absent.
    """
    prefix = f"{PROJECT_NAME}."
    return module_path[len(prefix) :] if module_path.startswith(prefix) else module_path


def _docname_from_module(module_path: str) -> str:
    """Convert a dotted module path to a slash-separated doc name.

    Args:
        module_path: Dotted Python module path (e.g. ``"easydel.kernels.foo"``).

    Returns:
        str: Slash-separated doc name relative to the API root (``"kernels/foo"``).
    """
    return _strip_prefix(module_path).replace(".", "/")


def create_rst_file(name: str, module_path: str, output_dir: Path) -> None:
    """Write one Sphinx ``automodule`` page for a Python module.

    The output path mirrors the package layout so the per-package
    ``index.rst`` toctrees can reference it without further translation.

    Args:
        name: Page title (typically the full dotted module path).
        module_path: Dotted module path to expand via ``automodule``.
        output_dir: Root directory holding the generated ``.rst`` tree.

    Returns:
        None.
    """
    docname = _docname_from_module(module_path)
    rst_path = output_dir / f"{docname}.rst"
    rst_path.parent.mkdir(parents=True, exist_ok=True)

    title = name
    with open(rst_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write(f".. automodule:: {module_path}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")


def _write_package_index(pkg_docname: str, children: list[str], packages: set[str]) -> None:
    """Write the ``index.rst`` toctree page for one package (or the API root).

    Packages are listed before modules; within each group entries are sorted
    alphabetically. Package entries become ``<child>/index`` toctree lines,
    module entries are referenced directly by their doc name.

    Args:
        pkg_docname: Slash-separated package doc name, or ``""`` for the
            top-level API root.
        children: Doc names of immediate children (packages and modules).
        packages: Set of all package doc names; used to detect which children
            should resolve to ``<child>/index`` toctree entries.

    Returns:
        None.
    """
    if pkg_docname:
        index_path = DOCS_API_DIR / pkg_docname / "index.rst"
    else:
        index_path = DOCS_API_DIR / "index.rst"

    index_path.parent.mkdir(parents=True, exist_ok=True)

    if pkg_docname:
        dotted = pkg_docname.replace("/", ".")
        title = f"{PROJECT_NAME}.{dotted} package"
    else:
        title = f"{PROJECT_NAME} API Reference"

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")

        for child in sorted(children, key=lambda c: (0 if c in packages else 1, c)):
            if pkg_docname:
                prefix = f"{pkg_docname}/"
                child_rel = child[len(prefix) :] if child.startswith(prefix) else child
            else:
                child_rel = child

            entry = f"{child_rel}/index" if child in packages else child_rel
            f.write(f"   {entry}\n")


def generate_api_docs(clean: bool = True) -> bool:
    """Discover every module under the package and emit a hierarchical Sphinx tree.

    Walks ``easydel/``, registers each non-``__init__`` module, builds the
    package -> child relationship map, and writes one ``automodule`` page per
    module plus one index per package and a top-level index.

    Args:
        clean: When ``True`` (default), recursively remove ``docs/api_docs/``
            before regenerating so stale pages are not left behind.

    Returns:
        bool: ``True`` on success; ``False`` when no modules were discovered.
    """
    print("Generating API documentation...")

    # Clean output dir completely to avoid stale files and empty dirs
    if clean and DOCS_API_DIR.exists():
        shutil.rmtree(DOCS_API_DIR)
    DOCS_API_DIR.mkdir(parents=True, exist_ok=True)

    # Discover modules (full import paths, e.g., easydel.kernels.foo.bar)
    modules = sorted(discover_modules(PROJECT_NAME))
    if not modules:
        print("No modules found to document")
        return False

    # Build package tree: packages set and a children map for toctrees
    packages: set[str] = set()
    children_map: defaultdict[str, set[str]] = defaultdict(set)

    for full_module in modules:
        short = _strip_prefix(full_module)
        parts = short.split(".")
        children_map[""].add(parts[0])

        # Register all ancestor packages
        for i in range(1, len(parts)):
            pkg = "/".join(parts[:i])
            packages.add(pkg)

        # Link package -> subpackage chain
        for i in range(1, len(parts) - 1):
            parent_pkg = "/".join(parts[:i])
            child_pkg = "/".join(parts[: i + 1])
            children_map[parent_pkg].add(child_pkg)

        # Add module as a child of its immediate package (or root if top-level)
        parent_pkg = "/".join(parts[:-1]) if len(parts) > 1 else ""
        mod_doc = "/".join(parts)
        children_map[parent_pkg].add(mod_doc)

    # Write module pages
    for module_path in modules:
        create_rst_file(module_path, module_path, DOCS_API_DIR)  # title = full import path

    # Write per-package index pages
    for pkg_docname in sorted(packages):
        children = list(children_map.get(pkg_docname, []))
        _write_package_index(pkg_docname, children, packages)

    # Write top-level index
    _write_package_index("", list(children_map[""]), packages)

    print(f"✓ Generated documentation for {len(modules)} modules")
    return True


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess | subprocess.CalledProcessError:
    """Run a subprocess command, capturing stdout/stderr.

    When the command fails and ``check`` is ``True`` the exception is logged
    and re-raised; when ``check`` is ``False`` the
    :class:`subprocess.CalledProcessError` is returned instead so the caller
    can inspect it.

    Args:
        cmd: Command and arguments as a list of strings.
        check: When ``True``, propagate non-zero exit codes as exceptions.

    Returns:
        subprocess.CompletedProcess | subprocess.CalledProcessError: The
            completed-process object on success, or the caught error when
            ``check`` is ``False`` and the command failed.

    Raises:
        subprocess.CalledProcessError: If ``check`` is ``True`` and the command
            exits non-zero.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def format_code(directory: str = PROJECT_NAME, fix: bool = True) -> bool:
    """Format every Python file under ``directory`` with ruff.

    When ``fix`` is ``True`` runs ``ruff check --fix --unsafe-fixes`` first,
    then unconditionally runs ``ruff format``. Both commands are pinned to
    the repo-root ``pyproject.toml`` configuration.

    Args:
        directory: Directory to format (defaults to the package name).
        fix: Whether to apply ``ruff check --fix`` before formatting.

    Returns:
        bool: ``True`` if both ruff invocations exited zero.
    """
    print(f"Formatting code in {directory}/...")

    # Get all Python files
    python_files = list(Path(directory).rglob("*.py"))

    if not python_files:
        print("No Python files found.")
        return True

    success = True

    # Run ruff check with optional fixes
    if fix:
        print("Running ruff check with fixes...")
        result = run_command(
            ["ruff", "check", "--fix", "--unsafe-fixes", "--config", "pyproject.toml"] + [str(f) for f in python_files],
            check=False,
        )
        if result.returncode != 0:
            print(f"Ruff check found issues (exit code: {result.returncode})")
            success = False

    # Run ruff format
    print("Running ruff format...")
    result = run_command(["ruff", "format", "--config", "pyproject.toml"] + [str(f) for f in python_files], check=False)
    if result.returncode != 0:
        print(f"Ruff format failed (exit code: {result.returncode})")
        success = False

    if success:
        print(f"✓ Successfully formatted {len(python_files)} files")
    else:
        print("✗ Some files had formatting issues")

    return success


def discover_modules(project_name: str) -> list[str]:
    """Walk a package directory and return every importable dotted module path.

    ``__init__.py`` files are skipped because Sphinx ``automodule`` already
    handles package-level docstrings via the per-package ``index.rst``.

    Args:
        project_name: Top-level package directory name under ``PROJECT_ROOT``.

    Returns:
        list[str]: Sorted, deduplicated dotted module paths
            (e.g. ``"easydel.kernels.flash_attention"``).

    Raises:
        FileNotFoundError: If ``PROJECT_ROOT/project_name`` does not exist.
    """
    base_dir = (PROJECT_ROOT / project_name).resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Package directory not found: {base_dir}")

    modules = []
    for py_file in base_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        rel = py_file.relative_to(base_dir)  # relative inside the package
        dotted = rel.with_suffix("").as_posix().replace("/", ".")
        modules.append(f"{project_name}.{dotted}")
    return sorted(set(modules))


def run_tests(test_dir: str = "test") -> bool:
    """Run the project's pytest suite under ``test_dir``.

    Args:
        test_dir: Directory containing tests (default ``"test"``).

    Returns:
        bool: ``True`` if pytest exited zero, otherwise ``False``.
    """
    print(f"Running tests in {test_dir}/...")

    result = run_command(["pytest", test_dir, "-v"], check=False)

    if result.returncode == 0:
        print("✓ All tests passed")
        return True
    else:
        print("✗ Some tests failed")
    return False


@dataclass
class ScriptArgs:
    """CLI arguments for :func:`main`.

    Attributes:
        format: Run ruff formatting.
        docs: Regenerate Sphinx API docs.
        test: Run the pytest suite.
        all: Shortcut for selecting every task.
        fix: Apply ruff's auto-fixes when formatting.
        clean: Remove ``docs/api_docs/`` before regenerating.
        directory: Directory to format with ruff.
    """

    format: bool = field(default=False, metadata={"help": "Format code with ruff"})
    docs: bool = field(default=False, metadata={"help": "Generate API documentation"})
    test: bool = field(default=False, metadata={"help": "Run tests"})
    all: bool = field(default=False, metadata={"help": "Run all tasks"})

    fix: bool = field(
        default=True,
        metadata={"help": "Apply fixes automatically (use --no-fix to disable)"},
    )
    clean: bool = field(
        default=True,
        metadata={"help": "Clean old documentation (use --no-clean to disable)"},
    )
    directory: str = field(default=PROJECT_NAME, metadata={"help": f"Directory to format (default: {PROJECT_NAME})"})


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the format/docs/test driver.

    When none of ``--format`` / ``--docs`` / ``--test`` is passed, runs
    formatting and documentation generation (i.e. the pre-commit shape).
    Raises ``SystemExit`` with code ``1`` when any selected step fails.

    Args:
        argv: Optional list of CLI tokens. ``None`` reads from ``sys.argv``.

    Returns:
        None.

    Raises:
        SystemExit: Always; the code reflects step success.
    """
    parser = DataClassArgumentParser(
        ScriptArgs,
        description=f"Format code and generate documentation for {PROJECT_NAME}",
    )
    (args,) = parser.parse_args_into_dataclasses(args=argv, look_for_args_file=False)

    run_all = args.all or not any([args.format, args.docs, args.test])
    exit_code = 0

    if run_all or args.format:
        if not format_code(args.directory, fix=args.fix):
            exit_code = 1

    if run_all or args.docs:
        if not generate_api_docs(clean=args.clean):
            exit_code = 1

    if args.test:
        if not run_tests():
            exit_code = 1

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
