import threading

from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.mixins.io import EngineIOMixin


class _DummyIOEngine(EngineIOMixin):
    def __init__(self, request_output: RequestOutput, *, scheduler_request_ids: set[str], active_request_ids: set[str]):
        self._output_lock = threading.RLock()
        self._scheduler_lock = threading.RLock()
        self._request_lock = threading.RLock()
        self._request_outputs = {request_output.request_id: request_output}
        self._active_requests = {rid: {} for rid in active_request_ids}
        self.scheduler = type("Scheduler", (), {"requests": {rid: object() for rid in scheduler_request_ids}})()
        self.aborted: list[str] = []

    def abort_request(self, request_id: str) -> None:
        self.aborted.append(request_id)
        with self._output_lock:
            ro = self._request_outputs.get(request_id)
            if ro is None:
                return
            for comp in ro.outputs:
                if comp.finish_reason is None:
                    comp.finish_reason = "abort"
            ro.finished = True
            ro.update_seq += 1


def _make_request_output(request_id: str, n: int, finished_indices: set[int] | None = None) -> RequestOutput:
    finished_indices = finished_indices or set()
    outputs = [CompletionOutput(index=i, text="", token_ids=[]) for i in range(n)]
    for i in finished_indices:
        outputs[i].finish_reason = "stop"
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=[],
        outputs=outputs,
        finished=False,
    )


def test_recover_orphaned_request_aborts_parent_when_children_disappear():
    ro = _make_request_output("req", n=4, finished_indices={0})
    engine = _DummyIOEngine(ro, scheduler_request_ids=set(), active_request_ids={"req"})

    recovered = engine._recover_orphaned_request("req")

    assert recovered is True
    assert engine.aborted == ["req"]
    assert ro.finished is True
    assert all(comp.finish_reason is not None for comp in ro.outputs)


def test_recover_orphaned_request_skips_when_scheduler_still_has_pending_child():
    ro = _make_request_output("req", n=4, finished_indices={0})
    engine = _DummyIOEngine(ro, scheduler_request_ids={"req-2"}, active_request_ids={"req", "req-2"})

    recovered = engine._recover_orphaned_request("req")

    assert recovered is False
    assert engine.aborted == []
    assert ro.finished is False
    assert ro.outputs[2].finish_reason is None


def test_recover_orphaned_request_handles_single_sample_requests():
    ro = _make_request_output("req-single", n=1)
    engine = _DummyIOEngine(ro, scheduler_request_ids=set(), active_request_ids=set())

    recovered = engine._recover_orphaned_request("req-single")

    assert recovered is True
    assert engine.aborted == ["req-single"]
    assert ro.finished is True
    assert ro.outputs[0].finish_reason == "abort"
