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

    roots = (
        Path("tests/trainers/runtime_pass"),
        Path("tests/trainers/mpmd_runtime_pass"),
    )
    modules: list[str] = []
    for root in roots:
        for path in sorted(root.glob("*.py")):
            if path.name != "__init__.py":
                modules.append(".".join(path.with_suffix("").parts))

    assert modules
    for module in modules:
        importlib.import_module(module)
