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

"""Building a model config does not start a JAX backend (dp replicas pick their JAX world after config loading)."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_ok

_SCRIPT = """
from jax._src import xla_bridge
import easydel as ed
from easydel.infra.base_config import EasyDeLBaseConfig
config = EasyDeLBaseConfig()
print(xla_bridge.backends_are_initialized(), config.backend)
"""


def _run(platforms: str) -> list[str]:
    env = {**os.environ, "JAX_PLATFORMS": platforms, "ENABLE_DISTRIBUTED_INIT": "0"}
    env.pop("XLA_FLAGS", None)
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=300, check=True
    )
    return out.stdout.strip().splitlines()[-1].split()


def test_config_init_takes_the_backend_name_from_jax_platforms_without_starting_it():
    assert _run("cpu") == ["False", "cpu"]


def test_config_init_names_the_first_configured_platform():
    # "tpu" is never started here (it would fail on a CPU-only host): only its name is read.
    assert _run("tpu,cpu") == ["False", "tpu"]
