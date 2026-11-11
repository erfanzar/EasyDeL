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

"""SIGTERM ends a serving rank through the engine's close(), including after uvicorn re-raises the signal."""

from __future__ import annotations

import os
import signal

import pytest
from easydel.inference.esurge.serving import app

pytestmark = pytest.mark.cpu_ok


def test_sigterm_raises_inside_the_context_and_the_previous_handler_comes_back():
    previous = signal.getsignal(signal.SIGTERM)
    with app.sigterm_raises():
        with pytest.raises(app.ShutdownRequested, match="SIGTERM"):
            os.kill(os.getpid(), signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) is previous


def test_a_signal_re_raised_after_a_server_restores_handlers_still_unwinds():
    """Uvicorn's capture_signals: swap in its own handler, restore the previous one, then raise_signal()."""
    seen = []
    with app.sigterm_raises():
        with pytest.raises(app.ShutdownRequested):
            restored = signal.signal(signal.SIGTERM, lambda s, f: seen.append(s))
            os.kill(os.getpid(), signal.SIGTERM)  # the server's graceful shutdown
            signal.signal(signal.SIGTERM, restored)
            signal.raise_signal(signal.SIGTERM)
    assert seen == [signal.SIGTERM]


def test_serve_closes_the_engine_and_returns_on_sigterm(monkeypatch):
    closed = []

    class Engine:
        is_coordination_leader = True

        def close(self):
            closed.append(True)

    def fake_serve(config):
        engine = Engine()
        try:
            os.kill(os.getpid(), signal.SIGTERM)  # arrives while the leader serves HTTP
        finally:
            engine.close()

    monkeypatch.setattr(app, "_serve", fake_serve)
    app.serve(config=None)
    assert closed == [True]
