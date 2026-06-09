import json
import os
import subprocess
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "eformer").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from: {start}")


def _discover_modules(repo_root: Path) -> list[str]:
    pkg_root = repo_root / "eformer"
    modules: list[str] = []
    for path in pkg_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(repo_root)
        if path.name == "__init__.py":
            mod = ".".join(rel.parent.parts)
        else:
            mod = ".".join(rel.with_suffix("").parts)
        modules.append(mod)
    return sorted(set(modules))


def _extract_json(stdout: str) -> dict:
    lines = stdout.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise AssertionError("Subprocess did not return JSON output.")


def test_imports_do_not_initialize_xla() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve())
    modules = _discover_modules(repo_root)

    code = """
import importlib
import json
import sys

modules = json.loads(sys.argv[1])
result = {
    "modules_tested": len(modules),
    "triggered": [],
    "import_errors": {},
    "xla_initialized_end": False,
}

xla_initialized = False
for mod in modules:
    try:
        importlib.import_module(mod)
    except Exception as e:
        result["import_errors"][mod] = repr(e)
        continue

    if "jax" not in sys.modules:
        continue

    try:
        from jax._src import xla_bridge
        now = bool(xla_bridge.backends_are_initialized())
    except Exception as e:
        result["import_errors"][mod] = f"xla_bridge_error: {repr(e)}"
        continue

    if not xla_initialized and now:
        result["triggered"].append(mod)
        xla_initialized = True

result["xla_initialized_end"] = xla_initialized
print(json.dumps(result))
""".strip()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    proc = subprocess.run(
        [sys.executable, "-c", code, json.dumps(modules)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, f"Subprocess failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"

    data = _extract_json(proc.stdout)
    import_errors = data.get("import_errors", {})
    triggered = data.get("triggered", [])

    assert not import_errors, f"Module import errors: {import_errors}"
    assert not triggered, f"Modules initialized XLA backends on import: {triggered}"
