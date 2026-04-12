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


def test_import_easydel_makes_eformer_exceptioninfo_safe_for_retryerror():
    env = os.environ.copy()
    env["ENABLE_DISTRIBUTED_INIT"] = "0"
    code = """
import pickle

import easydel
from eformer.executor.ray.types import ExceptionInfo
from google.api_core.exceptions import RetryError

try:
    raise RetryError("boom", RuntimeError("inner"))
except RetryError as exc:
    info = ExceptionInfo.ser_exc_info(exc)

payload = pickle.dumps(info)
restored = pickle.loads(payload)
print(type(restored.ex).__name__)
print(str(restored.ex))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert lines[-2] == "RuntimeError"
    assert lines[-1] == "google.api_core.exceptions.RetryError: boom, last exception: inner"
