from __future__ import annotations

import os
import subprocess
import sys


def _run_import_probe(env_overrides: dict[str, str] | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    code = (
        "from jax._src import xla_bridge\n"
        "print(xla_bridge.backends_are_initialized())\n"
        "import easydel\n"
        "print(xla_bridge.backends_are_initialized())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_import_easydel_does_not_initialize_jax_backends_by_default():
    returncode, stdout, _stderr = _run_import_probe()
    assert returncode == 0
    lines = [line.strip() for line in stdout.splitlines() if line.strip() in {"True", "False"}]
    assert lines[:2] == ["False", "False"]


def test_import_easydel_with_distributed_init_disabled_stays_lazy():
    returncode, stdout, _stderr = _run_import_probe({"ENABLE_DISTRIBUTED_INIT": "0"})
    assert returncode == 0
    lines = [line.strip() for line in stdout.splitlines() if line.strip() in {"True", "False"}]
    assert lines[:2] == ["False", "False"]
