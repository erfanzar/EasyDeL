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

from easydel.workers.esurge.pipeline.worker_manager import DEFAULT_WORKER_STARTUP_TIMEOUT, WorkerManager


def test_worker_manager_reads_startup_timeout_from_env(monkeypatch):
    monkeypatch.setenv("EASURGE_WORKER_STARTUP_TIMEOUT", "77.5")

    manager = WorkerManager("dummy-tokenizer")

    assert manager._startup_timeout == 77.5


def test_worker_manager_prefers_explicit_startup_timeout(monkeypatch):
    monkeypatch.setenv("EASURGE_WORKER_STARTUP_TIMEOUT", "77.5")

    manager = WorkerManager("dummy-tokenizer", startup_timeout=12.0)

    assert manager._startup_timeout == 12.0


def test_worker_manager_ignores_invalid_env_timeout(monkeypatch):
    monkeypatch.setenv("EASURGE_WORKER_STARTUP_TIMEOUT", "not-a-number")

    manager = WorkerManager("dummy-tokenizer")

    assert manager._startup_timeout == DEFAULT_WORKER_STARTUP_TIMEOUT
