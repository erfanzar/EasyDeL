#!/usr/bin/env python3
"""
Standalone script to format code and generate API documentation for easydel.
This works as a pre-commit hook when called from .pre-commit-config.yaml.
"""

import argparse
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_NAME = "easydel"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_API_DIR = PROJECT_ROOT / "docs" / "api_docs"


def _strip_prefix(module_path: str) -> str:
    prefix = f"{PROJECT_NAME}."
    return module_path[len(prefix) :] if module_path.startswith(prefix) else module_path


def _docname_from_module(module_path: str) -> str:
    return _strip_prefix(module_path).replace(".", "/")


def create_rst_file(name: str, module_path: str, output_dir: Path) -> None:
    """
    Create a module page at a nested path (mirrors package structure).
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
    """
    Write index.rst for a package.
    pkg_docname: '' for top-level, otherwise 'kernels' or 'kernels/triton_flash_attention', etc.
    children: list of docnames for immediate children (packages or modules)
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

        # Show packages first, then modules
        for child in sorted(children, key=lambda c: (0 if c in packages else 1, c)):
            entry = f"{child}/index" if child in packages else child
            f.write(f"   {entry}\n")


def generate_api_docs(clean: bool = True) -> bool:
    """
    Generate API documentation with a hierarchical layout.
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


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the result.

    Args:
        cmd: Command and arguments as list.
        check: Whether to raise exception on non-zero exit code.

    Returns:
        CompletedProcess object with command results.
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
    """
    Format Python code using ruff.

    Args:
        directory: Directory to format.
        fix: Whether to apply fixes automatically.

    Returns:
        True if successful, False otherwise.
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
    """
    Run project tests using pytest.

    Args:
        test_dir: Directory containing tests.

    Returns:
        True if tests pass, False otherwise.
    """
    print(f"Running tests in {test_dir}/...")

    result = run_command(["pytest", test_dir, "-v"], check=False)

    if result.returncode == 0:
        print("✓ All tests passed")
        return True
    else:
        print("✗ Some tests failed")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=f"Format code and generate documentation for {PROJECT_NAME}")

    # Task selection
    parser.add_argument("--format", action="store_true", help="Format code with ruff")
    parser.add_argument("--docs", action="store_true", help="Generate API documentation")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--all", action="store_true", help="Run all tasks")

    # Options
    parser.add_argument(
        "--no-fix",
        dest="fix",
        action="store_false",
        default=True,
        help="Don't apply fixes automatically",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        default=True,
        help="Don't clean old documentation",
    )
    parser.add_argument(
        "--directory",
        default=PROJECT_NAME,
        help=f"Directory to format (default: {PROJECT_NAME})",
    )

    args = parser.parse_args()

    # Default to all if no specific task selected
    if not any([args.format, args.docs, args.test, args.all]):
        args.all = True

    exit_code = 0

    # Run selected tasks
    if args.all or args.format:
        if not format_code(args.directory, fix=args.fix):
            exit_code = 1

    if args.all or args.docs:
        if not generate_api_docs(clean=args.clean):
            exit_code = 1

    if args.test:
        if not run_tests():
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
