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


def _run_libtpu_env_probe(env_overrides: dict[str, str] | None = None, code: str | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    env.pop("LIBTPU_INIT_ARGS", None)
    env.pop("EASYDEL_TARGETED_TPU_GENERATION", None)
    env.pop("TPU_ACCELERATOR_TYPE", None)
    env.pop("TPU_TYPE", None)
    env.pop("TPU_VERSION", None)
    env.pop("ACCELERATOR_TYPE", None)
    if env_overrides:
        env.update(env_overrides)
    code = code or "import os\nimport easydel\nprint(os.environ.get('LIBTPU_INIT_ARGS', ''))\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_transformers_mrope_validation_probe() -> tuple[int, str, str]:
    code = (
        "import logging\n"
        "import easydel\n"
        "from transformers.configuration_utils import PretrainedConfig\n"
        "logging.basicConfig(level=logging.WARNING)\n"
        "cfg = PretrainedConfig(\n"
        "    rope_parameters={\n"
        "        'rope_type': 'default',\n"
        "        'rope_theta': 10000.0,\n"
        "        'mrope_section': [24, 20, 20],\n"
        "        'mrope_interleaved': True,\n"
        "    }\n"
        ")\n"
        "cfg.validate_rope()\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ.copy(), "ENABLE_DISTRIBUTED_INIT": "0"},
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _last_stdout_line(stdout: str) -> str:
    lines = stdout.splitlines()
    return lines[-1] if lines else ""


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


def test_import_easydel_tpu_generation_env_detection_stays_jax_lazy():
    returncode, stdout, _stderr = _run_import_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "TPU_TYPE": "v6e-8",
        }
    )
    assert returncode == 0
    lines = [line.strip() for line in stdout.splitlines() if line.strip() in {"True", "False"}]
    assert lines[:2] == ["False", "False"]


def test_import_easydel_sets_preemption_service_env_default():
    returncode, stdout, _stderr = _run_preemption_env_probe({"ENABLE_DISTRIBUTED_INIT": "0"})
    assert returncode == 0
    assert stdout.strip().splitlines()[-1] == "true"


def test_import_easydel_detects_tpu_generation_from_env_without_explicit_selector():
    returncode, stdout, _stderr = _run_libtpu_env_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "TPU_TYPE": "v6e-8",
        }
    )
    assert returncode == 0
    flags = _last_stdout_line(stdout)
    assert "--xla_tpu_scoped_vmem_limit_kib=98304" in flags
    assert "--xla_tpu_enable_async_collective_fusion=true" in flags


def test_import_easydel_exports_detected_tpu_generation():
    returncode, stdout, _stderr = _run_libtpu_env_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "TPU_TYPE": "v6e-8",
        },
        code=("import os\nimport easydel\nprint(os.environ.get('EASYDEL_TARGETED_TPU_GENERATION', ''))\n"),
    )
    assert returncode == 0
    assert _last_stdout_line(stdout) == "v6e"


def test_import_easydel_applies_maxtext_v6e_libtpu_flags_when_requested():
    returncode, stdout, _stderr = _run_libtpu_env_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "EASYDEL_TARGETED_TPU_GENERATION": "max-6",
        }
    )
    assert returncode == 0
    flags = _last_stdout_line(stdout)
    assert "--xla_tpu_scoped_vmem_limit_kib=98304" in flags
    assert "--xla_tpu_enable_async_collective_fusion=true" in flags
    assert "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true" in flags
    assert "TPU_MEGACORE=MEGACORE_DENSE" not in flags


def test_import_easydel_ignores_unknown_tpu_generation():
    returncode, stdout, _stderr = _run_libtpu_env_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "EASYDEL_TARGETED_TPU_GENERATION": "v7x",
        }
    )
    assert returncode == 0
    baseline_returncode, baseline_stdout, _baseline_stderr = _run_libtpu_env_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "EASYDEL_AUTO": "0",
        }
    )
    assert baseline_returncode == 0
    assert _last_stdout_line(stdout) == _last_stdout_line(baseline_stdout)


def test_import_easydel_silences_hf_default_rope_mrope_metadata_warning():
    returncode, stdout, stderr = _run_transformers_mrope_validation_probe()
    assert returncode == 0
    assert _last_stdout_line(stdout) == "ok"
    assert "Unrecognized keys in `rope_parameters`" not in stderr
    assert "mrope_section" not in stderr
    assert "mrope_interleaved" not in stderr
