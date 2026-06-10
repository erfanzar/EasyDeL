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
"""Format workspace packages with ruff and (re)generate their Sphinx API docs.

One driver for every package in the workspace (``libs/easydel``,
``libs/spectrax``, ``libs/ejkernel``, ``libs/eformer``). For each selected
library it runs up to three steps:

1. ``format_code`` — ``ruff check --fix`` (when ``--fix``) and ``ruff format``
   over the library's package directory, using that library's own
   ``pyproject.toml`` ruff configuration.
2. ``generate_api_docs`` — walks the package, emits one ``.rst`` page per
   module mirroring the package layout under ``libs/<lib>/docs/api_docs/``,
   and writes per-package and root index pages with sorted toctree entries.
3. ``run_tests`` — runs the library's pytest suite when ``--test`` is set.

By default (no flags), formatting and docs run for ``easydel`` only.

Side effects:
    - Mutates files in place under the package directory via ruff fixes.
    - Removes and recreates ``libs/<lib>/docs/api_docs/`` (when ``--clean``).

Usage:
    python scripts/format_and_generate_docs.py                  # easydel: format + docs
    python scripts/format_and_generate_docs.py --libs all       # every workspace package
    python scripts/format_and_generate_docs.py --libs ejkernel,eformer --docs
    python scripts/format_and_generate_docs.py --all --libs spectrax
"""

import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from eformer.aparser import DataClassArgumentParser

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Lib:
    """One workspace package the driver can operate on.

    Attributes:
        name: Directory name under ``libs/`` (also the import package name).
        tests_dir: Test directory name inside the library root (``"tests"``
            for most libraries, ``"test"`` for ejkernel).
    """

    name: str
    tests_dir: str = "tests"

    @property
    def root(self) -> Path:
        """Library project root (``libs/<name>``)."""
        return WORKSPACE_ROOT / "libs" / self.name

    @property
    def package_dir(self) -> Path:
        """Import-package directory (``libs/<name>/<name>``)."""
        return self.root / self.name

    @property
    def docs_api_dir(self) -> Path:
        """Generated Sphinx API tree (``libs/<name>/docs/api_docs``)."""
        return self.root / "docs" / "api_docs"

    @property
    def ruff_config(self) -> Path:
        """The library's own pyproject, holding its ruff configuration."""
        return self.root / "pyproject.toml"


LIBS: dict[str, Lib] = {
    "easydel": Lib("easydel"),
    "spectrax": Lib("spectrax"),
    "ejkernel": Lib("ejkernel", tests_dir="test"),
    "eformer": Lib("eformer"),
}


def _strip_prefix(module_path: str, project_name: str) -> str:
    """Strip the top-level ``"<project_name>."`` prefix from a dotted module path.

    Args:
        module_path: Dotted Python module path.
        project_name: Top-level package name to strip.

    Returns:
        str: ``module_path`` with the leading prefix removed, unchanged when
            the prefix is absent.
    """
    prefix = f"{project_name}."
    return module_path[len(prefix) :] if module_path.startswith(prefix) else module_path


def _docname_from_module(module_path: str, project_name: str) -> str:
    """Convert a dotted module path to a slash-separated doc name.

    Args:
        module_path: Dotted Python module path (e.g. ``"ejkernel.kernels.foo"``).
        project_name: Top-level package name to strip.

    Returns:
        str: Slash-separated doc name relative to the API root (``"kernels/foo"``).
    """
    return _strip_prefix(module_path, project_name).replace(".", "/")


def create_rst_file(name: str, module_path: str, output_dir: Path, project_name: str) -> None:
    """Write one Sphinx ``automodule`` page for a Python module.

    The output path mirrors the package layout so the per-package
    ``index.rst`` toctrees can reference it without further translation.

    Args:
        name: Page title (typically the full dotted module path).
        module_path: Dotted module path to expand via ``automodule``.
        output_dir: Root directory holding the generated ``.rst`` tree.
        project_name: Top-level package name used to relativize paths.

    Returns:
        None.
    """
    docname = _docname_from_module(module_path, project_name)
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


def _write_package_index(
    pkg_docname: str,
    children: list[str],
    packages: set[str],
    docs_api_dir: Path,
    project_name: str,
) -> None:
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
        docs_api_dir: Root of the generated API doc tree.
        project_name: Top-level package name used in page titles.

    Returns:
        None.
    """
    if pkg_docname:
        index_path = docs_api_dir / pkg_docname / "index.rst"
    else:
        index_path = docs_api_dir / "index.rst"

    index_path.parent.mkdir(parents=True, exist_ok=True)

    if pkg_docname:
        dotted = pkg_docname.replace("/", ".")
        title = f"{project_name}.{dotted} package"
    else:
        title = f"{project_name} API Reference"

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


def generate_api_docs(lib: Lib, clean: bool = True) -> bool:
    """Discover every module under a library's package and emit its Sphinx tree.

    Walks ``libs/<lib>/<lib>/``, registers each non-``__init__`` module,
    builds the package -> child relationship map, and writes one
    ``automodule`` page per module plus one index per package and a top-level
    index under ``libs/<lib>/docs/api_docs/``.

    Args:
        lib: The workspace library to document.
        clean: When ``True`` (default), recursively remove the library's
            ``docs/api_docs/`` before regenerating so stale pages are not
            left behind.

    Returns:
        bool: ``True`` on success; ``False`` when no modules were discovered.
    """
    print(f"[{lib.name}] Generating API documentation...")

    docs_api_dir = lib.docs_api_dir
    if clean and docs_api_dir.exists():
        shutil.rmtree(docs_api_dir)
    docs_api_dir.mkdir(parents=True, exist_ok=True)

    modules = sorted(discover_modules(lib))
    if not modules:
        print(f"[{lib.name}] No modules found to document")
        return False

    packages: set[str] = set()
    children_map: defaultdict[str, set[str]] = defaultdict(set)

    for full_module in modules:
        short = _strip_prefix(full_module, lib.name)
        parts = short.split(".")
        children_map[""].add(parts[0])

        for i in range(1, len(parts)):
            pkg = "/".join(parts[:i])
            packages.add(pkg)

        for i in range(1, len(parts) - 1):
            parent_pkg = "/".join(parts[:i])
            child_pkg = "/".join(parts[: i + 1])
            children_map[parent_pkg].add(child_pkg)

        parent_pkg = "/".join(parts[:-1]) if len(parts) > 1 else ""
        mod_doc = "/".join(parts)
        children_map[parent_pkg].add(mod_doc)

    for module_path in modules:
        create_rst_file(module_path, module_path, docs_api_dir, lib.name)

    for pkg_docname in sorted(packages):
        children = list(children_map.get(pkg_docname, []))
        _write_package_index(pkg_docname, children, packages, docs_api_dir, lib.name)

    _write_package_index("", list(children_map[""]), packages, docs_api_dir, lib.name)

    print(f"[{lib.name}] ✓ Generated documentation for {len(modules)} modules")
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


def format_code(lib: Lib, fix: bool = True) -> bool:
    """Format a library's package directory with ruff.

    When ``fix`` is ``True`` runs ``ruff check --fix --unsafe-fixes`` first,
    then unconditionally runs ``ruff format``. Both commands are pinned to
    the library's own ``pyproject.toml`` configuration.

    Args:
        lib: The workspace library to format.
        fix: Whether to apply ``ruff check --fix`` before formatting.

    Returns:
        bool: ``True`` if both ruff invocations exited zero.
    """
    print(f"[{lib.name}] Formatting code in {lib.package_dir.relative_to(WORKSPACE_ROOT)}/...")

    python_files = list(lib.package_dir.rglob("*.py"))
    if not python_files:
        print(f"[{lib.name}] No Python files found.")
        return True

    config = str(lib.ruff_config)
    success = True

    if fix:
        print(f"[{lib.name}] Running ruff check with fixes...")
        result = run_command(
            ["ruff", "check", "--fix", "--unsafe-fixes", "--config", config] + [str(f) for f in python_files],
            check=False,
        )
        if result.returncode != 0:
            print(f"[{lib.name}] Ruff check found issues (exit code: {result.returncode})")
            success = False

    print(f"[{lib.name}] Running ruff format...")
    result = run_command(["ruff", "format", "--config", config] + [str(f) for f in python_files], check=False)
    if result.returncode != 0:
        print(f"[{lib.name}] Ruff format failed (exit code: {result.returncode})")
        success = False

    if success:
        print(f"[{lib.name}] ✓ Successfully formatted {len(python_files)} files")
    else:
        print(f"[{lib.name}] ✗ Some files had formatting issues")

    return success


def discover_modules(lib: Lib) -> list[str]:
    """Walk a library's package directory and return every dotted module path.

    ``__init__.py`` files are skipped because Sphinx ``automodule`` already
    handles package-level docstrings via the per-package ``index.rst``.

    Args:
        lib: The workspace library to scan.

    Returns:
        list[str]: Sorted, deduplicated dotted module paths
            (e.g. ``"ejkernel.kernels.flash_attention"``).

    Raises:
        FileNotFoundError: If the package directory does not exist.
    """
    base_dir = lib.package_dir.resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Package directory not found: {base_dir}")

    modules = []
    for py_file in base_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        rel = py_file.relative_to(base_dir)
        dotted = rel.with_suffix("").as_posix().replace("/", ".")
        modules.append(f"{lib.name}.{dotted}")
    return sorted(set(modules))


def run_tests(lib: Lib) -> bool:
    """Run a library's pytest suite.

    Args:
        lib: The workspace library whose tests to run.

    Returns:
        bool: ``True`` if pytest exited zero, otherwise ``False``.
    """
    test_dir = lib.root / lib.tests_dir
    print(f"[{lib.name}] Running tests in {test_dir.relative_to(WORKSPACE_ROOT)}/...")

    result = run_command(["pytest", str(test_dir), "-v"], check=False)

    if result.returncode == 0:
        print(f"[{lib.name}] ✓ All tests passed")
        return True
    print(f"[{lib.name}] ✗ Some tests failed")
    return False


def _select_libs(spec: str) -> list[Lib]:
    """Resolve the ``--libs`` CLI value into library objects.

    Args:
        spec: ``"all"`` or a comma-separated subset of the workspace
            library names (e.g. ``"easydel"``, ``"ejkernel,eformer"``).

    Returns:
        list[Lib]: The selected libraries, in declaration order.

    Raises:
        SystemExit: If an unknown library name is given.
    """
    if spec.strip().lower() == "all":
        return list(LIBS.values())
    selected = []
    for name in spec.split(","):
        name = name.strip()
        if name not in LIBS:
            raise SystemExit(f"unknown lib: {name!r} (choose from {', '.join(LIBS)} or 'all')")
        selected.append(LIBS[name])
    return selected


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
        libs: Which workspace libraries to operate on.
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
    libs: str = field(
        default="easydel",
        metadata={"help": f"Comma-separated workspace libs or 'all' (choices: {', '.join(LIBS)})"},
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the per-library format/docs/test driver.

    When none of ``--format`` / ``--docs`` / ``--test`` is passed, runs
    formatting and documentation generation (i.e. the pre-commit shape) for
    each selected library. Raises ``SystemExit`` with code ``1`` when any
    selected step fails for any library.

    Args:
        argv: Optional list of CLI tokens. ``None`` reads from ``sys.argv``.

    Returns:
        None.

    Raises:
        SystemExit: Always; the code reflects step success.
    """
    parser = DataClassArgumentParser(
        ScriptArgs,
        description="Format code and generate documentation for the workspace libraries",
    )
    (args,) = parser.parse_args_into_dataclasses(args=argv, look_for_args_file=False)

    run_all = args.all or not any([args.format, args.docs, args.test])
    exit_code = 0

    for lib in _select_libs(args.libs):
        if run_all or args.format:
            if not format_code(lib, fix=args.fix):
                exit_code = 1

        if run_all or args.docs:
            if not generate_api_docs(lib, clean=args.clean):
                exit_code = 1

        if args.test:
            if not run_tests(lib):
                exit_code = 1

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
