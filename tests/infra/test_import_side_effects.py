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


def _run_preemption_env_probe(env_overrides: dict[str, str] | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    code = "import os\nimport easydel\nprint(os.environ.get('JAX_ENABLE_PREEMPTION_SERVICE'))\n"
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


def test_import_easydel_sets_preemption_service_env_default():
    returncode, stdout, _stderr = _run_preemption_env_probe({"ENABLE_DISTRIBUTED_INIT": "0"})
    assert returncode == 0
    assert stdout.strip().splitlines()[-1] == "true"
