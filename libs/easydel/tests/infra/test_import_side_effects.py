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

import pytest


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
    flags = _last_stdout_line(stdout).split()
    assert "--xla_tpu_scoped_vmem_limit_kib=98304" in flags
    assert "--xla_enable_async_all_gather=true" in flags
    assert "--xla_tpu_overlap_compute_collective_tc=true" in flags
    assert all(
        flag.split("=", 1)[0]
        not in {
            "--xla_tpu_enable_async_collective_fusion",
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather",
        }
        for flag in flags
    )


@pytest.mark.parametrize("selector", ["EASYDEL_TARGETED_TPU_GENERATION", "TPU_TYPE"])
def test_import_easydel_omits_v6e_continuation_fusion_flags(selector):
    returncode, stdout, stderr = _run_libtpu_env_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "XLA_FLAGS": "",
            selector: "v6e-4",
        },
        code=(
            "import os\n"
            "from jax._src import xla_bridge\n"
            "assert not xla_bridge.backends_are_initialized()\n"
            "import easydel\n"
            "assert not xla_bridge.backends_are_initialized()\n"
            "print(os.environ.get('LIBTPU_INIT_ARGS', ''))\n"
        ),
    )
    assert returncode == 0, stderr
    flags = _last_stdout_line(stdout).split()
    flag_names = {flag.split("=", 1)[0] for flag in flags}
    assert "--xla_tpu_enable_async_collective_fusion" not in flag_names
    assert "--xla_tpu_enable_async_collective_fusion_fuse_all_gather" not in flag_names
    assert "--xla_tpu_scoped_vmem_limit_kib=98304" in flags
    assert "--xla_enable_async_all_gather=true" in flags
    assert "--xla_tpu_overlap_compute_collective_tc=true" in flags
    assert "--xla_tpu_enable_async_collective_fusion_multiple_steps=true" in flags


@pytest.mark.parametrize("selector", ["EASYDEL_TARGETED_TPU_GENERATION", "TPU_TYPE"])
def test_import_easydel_disables_v5p_continuation_fusion_without_disabling_async_collectives(selector):
    returncode, stdout, stderr = _run_libtpu_env_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "XLA_FLAGS": "",
            selector: "v5p-8",
        },
        code=(
            "import os\n"
            "from jax._src import xla_bridge\n"
            "assert not xla_bridge.backends_are_initialized()\n"
            "import easydel\n"
            "assert not xla_bridge.backends_are_initialized()\n"
            "print(os.environ.get('LIBTPU_INIT_ARGS', ''))\n"
        ),
    )
    assert returncode == 0, stderr
    flags = _last_stdout_line(stdout).split()
    assert "--xla_tpu_enable_async_collective_fusion=false" in flags
    assert "--xla_tpu_enable_async_collective_fusion=true" not in flags
    assert "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false" in flags
    assert "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true" not in flags
    assert "--xla_enable_async_all_gather=true" in flags
    assert "--xla_enable_async_collective_permute=true" in flags
    assert "--xla_tpu_enable_ag_backward_pipelining=true" in flags


def test_import_easydel_keeps_v5e_continuation_fusion_defaults():
    returncode, stdout, stderr = _run_libtpu_env_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "XLA_FLAGS": "",
            "EASYDEL_TARGETED_TPU_GENERATION": "v5e",
        }
    )
    assert returncode == 0, stderr
    flags = _last_stdout_line(stdout).split()
    assert "--xla_tpu_enable_async_collective_fusion=true" in flags
    assert "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true" in flags
    assert "--xla_enable_async_all_gather=true" in flags


@pytest.mark.parametrize("source", ["LIBTPU_INIT_ARGS", "XLA_FLAGS"])
@pytest.mark.parametrize("generation,value", [("v5p", "true"), ("v5e", "false"), ("v6e", "false"), ("v6e", "true")])
def test_import_easydel_preserves_explicit_tpu_fusion_flags(source, generation, value):
    overrides = (
        f"--xla_tpu_enable_async_collective_fusion={value} "
        f"--xla_tpu_enable_async_collective_fusion_fuse_all_gather={value}"
    )
    returncode, stdout, stderr = _run_libtpu_env_probe(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "EASYDEL_TARGETED_TPU_GENERATION": generation,
            "XLA_FLAGS": "",
            source: overrides,
        },
        code=(
            "import os\n"
            "import easydel\n"
            f"assert os.environ[{source!r}].startswith({overrides!r})\n"
            "print(os.environ.get('LIBTPU_INIT_ARGS', ''))\n"
        ),
    )
    assert returncode == 0, stderr
    flags = _last_stdout_line(stdout).split()
    for name in ("--xla_tpu_enable_async_collective_fusion", "--xla_tpu_enable_async_collective_fusion_fuse_all_gather"):
        matching_flags = [flag for flag in flags if flag.split("=", 1)[0] == name]
        assert matching_flags == ([f"{name}={value}"] if source == "LIBTPU_INIT_ARGS" else [])
    assert "--xla_enable_async_all_gather=true" in flags


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
    flags = _last_stdout_line(stdout).split()
    flag_names = {flag.split("=", 1)[0] for flag in flags}
    assert "--xla_tpu_scoped_vmem_limit_kib=98304" in flags
    assert "--xla_enable_async_all_gather=true" in flags
    assert "--xla_tpu_overlap_compute_collective_tc=true" in flags
    assert "--xla_tpu_enable_async_collective_fusion_multiple_steps=true" in flags
    assert "--xla_tpu_enable_async_collective_fusion" not in flag_names
    assert "--xla_tpu_enable_async_collective_fusion_fuse_all_gather" not in flag_names
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


def test_import_easydel_does_not_import_eyvan_or_server_deps_or_spawn_threads():
    code = (
        "import sys, threading\n"
        "from jax._src import xla_bridge\n"
        "before_threads = set(threading.enumerate())\n"
        "import easydel\n"
        "after_threads = [t for t in threading.enumerate() if t not in before_threads and t.is_alive()]\n"
        "assert not xla_bridge.backends_are_initialized()\n"
        "assert after_threads == [], f'unexpected threads: {after_threads}'\n"
        "forbidden = ('eyvan', 'eyvan.sdk', 'eyvan.storage', 'eyvan.track', 'fastapi', 'uvicorn')\n"
        "loaded = [name for name in forbidden if name in sys.modules]\n"
        "assert loaded == [], f'unexpected modules loaded at import: {loaded}'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ.copy(), "ENABLE_DISTRIBUTED_INIT": "0"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert _last_stdout_line(proc.stdout) == "ok"


def test_easydel_top_level_excludes_eyvan_exports_and_internal_helpers_stay_lazy(tmp_path):
    cfg_path = tmp_path / "storage.json"
    code = (
        "import sys, threading\n"
        "from jax._src import xla_bridge\n"
        "import easydel as ed\n"
        "before_threads = set(threading.enumerate())\n"
        "removed_top_level_names = (\n"
        "    'PreemptionNotice',\n"
        "    'acknowledge_checkpoint',\n"
        "    'configure_storage',\n"
        "    'eyvan_execute',\n"
        "    'eyvan_init',\n"
        "    'get_monitoring_storage',\n"
        "    'get_storage',\n"
        "    'install_preemption_handler',\n"
        "    'monitoring_path',\n"
        "    'preemption_notice',\n"
        "    'reclaim_requested',\n"
        "    'resume_checkpoint_uri',\n"
        "    'run_on_tpu',\n"
        "    'storage_path',\n"
        ")\n"
        "for name in removed_top_level_names:\n"
        "    assert not hasattr(ed, name), f'top-level easydel must not export {name}'\n"
        "    assert name not in dir(ed), f'top-level easydel dir() must not include {name}'\n"
        "from easydel.utils import eyvan_helpers as eh\n"
        "for removed_dispatch in ('run_on_tpu', 'eyvan_execute', 'eyvan_init'):\n"
        "    assert not hasattr(eh, removed_dispatch), f'eyvan_helpers must not expose {removed_dispatch}'\n"
        "internal_names = (\n"
        "    'PreemptionNotice',\n"
        "    'acknowledge_checkpoint',\n"
        "    'configure_storage',\n"
        "    'get_monitoring_storage',\n"
        "    'get_storage',\n"
        "    'install_preemption_handler',\n"
        "    'is_eyvan_auto_track_enabled',\n"
        "    'monitoring_path',\n"
        "    'preemption_notice',\n"
        "    'reclaim_requested',\n"
        "    'resume_checkpoint_uri',\n"
        "    'set_eyvan_auto_track_enabled',\n"
        "    'storage_path',\n"
        ")\n"
        "for name in internal_names:\n"
        "    fn = getattr(eh, name)\n"
        "    assert callable(fn), f'{name} not callable'\n"
        "    assert fn.__module__ == 'easydel.utils.eyvan_helpers', f'{name} from {fn.__module__}'\n"
        "assert not xla_bridge.backends_are_initialized()\n"
        "forbidden_on_attr = ('eyvan', 'eyvan.sdk', 'eyvan.storage', 'eyvan.track', 'fastapi', 'uvicorn')\n"
        "loaded_attr = [m for m in forbidden_on_attr if m in sys.modules]\n"
        "assert loaded_attr == [], f'internal helper import loaded: {loaded_attr}'\n"
        "assert eh.preemption_notice(environment={}) is None\n"
        "assert eh.reclaim_requested(environment={}) is False\n"
        "assert eh.resume_checkpoint_uri(environment={}) is None\n"
        "forbidden_after_preempt = ('eyvan.sdk', 'fastapi', 'uvicorn')\n"
        "loaded_preempt = [m for m in forbidden_after_preempt if m in sys.modules]\n"
        "assert loaded_preempt == [], f'preemption helpers imported: {loaded_preempt}'\n"
        "class _StubBackend:\n"
        "    def bucket_exists(self, bucket): return True\n"
        "    def get_bucket(self, bucket): return None\n"
        "    def create_bucket(self, *a, **kw): raise AssertionError('create forbidden')\n"
        f"cfg = {str(cfg_path)!r}\n"
        "eh.configure_storage(\n"
        "    project='demo-proj',\n"
        "    region='us-central2',\n"
        "    bucket='eyvan_us_central2',\n"
        "    monitoring_bucket='eyvan_monitoring',\n"
        "    monitoring_location='us-central2',\n"
        "    create=False,\n"
        "    config_path=cfg,\n"
        ")\n"
        "uri = eh.storage_path('checkpoints', 'demo', region='us-central2', create=False, config_path=cfg, backend=_StubBackend())\n"
        "mon = eh.monitoring_path('runs', create=False, config_path=cfg, backend=_StubBackend())\n"
        "assert uri == 'gs://eyvan_us_central2/checkpoints/demo', uri\n"
        "assert mon == 'gs://eyvan_monitoring/tracking/runs', mon\n"
        "forbidden_after_storage = ('eyvan.sdk', 'fastapi', 'uvicorn')\n"
        "loaded_storage = [m for m in forbidden_after_storage if m in sys.modules]\n"
        "assert loaded_storage == [], f'storage helpers imported: {loaded_storage}'\n"
        "after_threads = [t for t in threading.enumerate() if t not in before_threads and t.is_alive()]\n"
        "assert after_threads == [], f'unexpected threads: {after_threads}'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ.copy(), "ENABLE_DISTRIBUTED_INIT": "0", "HOME": str(tmp_path)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert _last_stdout_line(proc.stdout) == "ok"
