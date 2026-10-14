from __future__ import annotations

import importlib
from pathlib import Path


def test_trainer_runtime_pass_modules_import() -> None:
    """Keep script-style trainer runtime smoke entrypoints importable.

    The files under ``runtime_pass`` and ``mpmd_runtime_pass`` are executable
    trainer smoke scripts rather than normal pytest modules. Importing them
    validates their shared helpers, trainer/config wiring, and reward-function
    dependencies without launching the expensive training loops in ``main``.
    """

    # Resolved against this file rather than the process CWD: globbing relative
    # paths silently yields nothing when pytest runs from the repository root,
    # which turns this into a vacuous pass or an `assert modules` failure
    # depending on where it was invoked from.
    tests_root = Path(__file__).resolve().parents[1]
    roots = (
        tests_root / "trainers" / "runtime_pass",
        tests_root / "trainers" / "mpmd_runtime_pass",
    )
    modules: list[str] = []
    for root in roots:
        for path in sorted(root.glob("*.py")):
            if path.name != "__init__.py":
                relative = path.relative_to(tests_root.parent).with_suffix("")
                modules.append(".".join(relative.parts))

    assert modules
    for module in modules:
        importlib.import_module(module)
