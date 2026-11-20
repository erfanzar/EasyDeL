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

"""The eLarge serve action reads the joined JAX world without starting a backend (dp replicas need it unstarted)."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_ok

_SCRIPT = """
from jax._src import xla_bridge
from jax._src.distributed import global_state
from easydel.scripts.elarge import _joined_world
print(_joined_world(), xla_bridge.backends_are_initialized())

class _Client:
    pass

global_state.client = _Client()
global_state.num_processes, global_state.process_id = 16, 5
print(_joined_world(), xla_bridge.backends_are_initialized())
global_state.client = None
"""


def test_joined_world_comes_from_distributed_state_and_starts_no_backend():
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "ENABLE_DISTRIBUTED_INIT": "0"}
    env.pop("XLA_FLAGS", None)
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=300, check=True
    )
    lines = out.stdout.strip().splitlines()[-2:]
    assert lines == ["(1, 0) False", "(16, 5) False"]
