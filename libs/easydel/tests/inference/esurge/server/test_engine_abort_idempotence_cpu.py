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

"""Pure-Python eSurge abort contracts; no arrays or model initialization."""

import threading
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.esurge_engine import eSurge

pytestmark = pytest.mark.cpu_ok


def _stub_engine(output, *, active=False):
    engine = object.__new__(eSurge)
    engine._scheduler_lock = threading.RLock()
    engine._request_lock = threading.RLock()
    engine._output_lock = threading.RLock()
    engine._active_requests = {"req": SimpleNamespace(parent_request_id=None, sample_index=0)} if active else {}
    engine._request_outputs = {"req": output}
    engine._request_events = {"req": threading.Event()}
    engine._output_event = threading.Event()
    engine._max_request_outputs = None
    engine._failed_detokenizer_resets = set()
    engine._detokenizer_cleanup_threshold = 100
    resets = []
    finished_calls = []
    notified = []
    engine._detokenizer_client = SimpleNamespace(reset=resets.append)
    engine._request_plane = SimpleNamespace(owns_scheduling=False, notify_abort=notified.append)
    requests = {"req": SimpleNamespace(parent_request_id=None)} if active else {}

    def finish_requests(ids, status):
        finished_calls.append((set(ids), status))
        for rid in ids:
            requests.pop(rid, None)

    engine.scheduler = SimpleNamespace(requests=requests, running=[], waiting=[], finish_requests=finish_requests)
    return engine, resets, finished_calls, notified


def test_abort_retained_finished_output_preserves_terminal_state():
    output = SimpleNamespace(finished=True, outputs=[SimpleNamespace(finish_reason="stop")], update_seq=7)
    engine, resets, finished_calls, notified = _stub_engine(output)

    engine.abort_request("req")
    engine.abort_request("req")

    # The finished output's terminal state is never rewritten.
    assert output.finished is True
    assert output.outputs[0].finish_reason == "stop"
    assert output.update_seq == 7
    # No live rows: nothing to finish and no detokenizer state to reset.
    assert finished_calls == []
    assert resets == []
    # The remote owner is always told (it may still hold rows); its abort is idempotent too.
    assert notified == ["req", "req"]


def test_abort_finished_output_with_live_scheduler_row_frees_row_and_keeps_stop():
    # A stop string marks the output finished before the scheduler drains the row.
    output = SimpleNamespace(finished=True, outputs=[SimpleNamespace(finish_reason="stop")], update_seq=3)
    engine, resets, finished_calls, notified = _stub_engine(output, active=True)

    engine.abort_request("req")

    assert len(finished_calls) == 1
    assert finished_calls[0][0] == {"req"}
    assert "req" not in engine.scheduler.requests
    assert resets == ["req"]
    assert notified == ["req"]
    assert output.finished is True
    assert output.outputs[0].finish_reason == "stop"
    assert output.update_seq == 3


def test_abort_finished_child_with_live_row_does_not_touch_live_sibling():
    # n=2: child 1 finished on a stop string and lost its active record, but its scheduler row is
    # still live; sibling 0 is still generating.
    output = SimpleNamespace(
        finished=False,
        outputs=[SimpleNamespace(finish_reason=None), SimpleNamespace(finish_reason="stop")],
        update_seq=5,
    )
    engine, resets, finished_calls, notified = _stub_engine(output)
    engine.scheduler.requests["req-0"] = SimpleNamespace(parent_request_id="req", sample_index=0)
    engine.scheduler.requests["req-1"] = SimpleNamespace(parent_request_id="req", sample_index=1)
    engine._active_requests["req-0"] = SimpleNamespace(parent_request_id="req", sample_index=0)

    engine.abort_request("req-1")

    assert [call[0] for call in finished_calls] == [{"req-1"}]
    assert "req-0" in engine.scheduler.requests
    assert "req-1" not in engine.scheduler.requests
    assert [item.finish_reason for item in output.outputs] == [None, "stop"]
    assert output.finished is False
    assert output.update_seq == 5
    assert resets == ["req-1"]
    assert notified == ["req-1"]


def test_late_abort_of_released_child_does_not_touch_live_sibling():
    # n=2: child 1 already finished and was fully released (no active record, no scheduler row) while
    # sibling 0 is still generating. A late abort for the child id must not mark sibling 0 aborted.
    output = SimpleNamespace(
        finished=False,
        outputs=[SimpleNamespace(finish_reason=None), SimpleNamespace(finish_reason="stop")],
        update_seq=5,
    )
    engine, _resets, finished_calls, _notified = _stub_engine(output)
    engine.scheduler.requests["req-0"] = SimpleNamespace(parent_request_id="req", sample_index=0)
    engine._active_requests["req-0"] = SimpleNamespace(parent_request_id="req", sample_index=0)

    engine.abort_request("req-1")

    assert finished_calls == []
    assert "req-0" in engine.scheduler.requests
    assert [item.finish_reason for item in output.outputs] == [None, "stop"]
    assert output.finished is False
    assert output.update_seq == 5
    assert "req-0" in engine._active_requests


def test_repeated_abort_of_unfinished_output_preserves_completed_sample():
    output = SimpleNamespace(
        finished=False,
        outputs=[SimpleNamespace(finish_reason="stop"), SimpleNamespace(finish_reason=None)],
        update_seq=0,
    )
    engine, resets, finished_calls, notified = _stub_engine(output, active=True)

    engine.abort_request("req")
    snapshot = (output.finished, [item.finish_reason for item in output.outputs], output.update_seq)
    engine.abort_request("req")

    assert snapshot == (True, ["stop", "abort"], 1)
    assert (output.finished, [item.finish_reason for item in output.outputs], output.update_seq) == snapshot
    assert len(finished_calls) == 1
    assert resets == ["req"]
    assert notified == ["req", "req"]
