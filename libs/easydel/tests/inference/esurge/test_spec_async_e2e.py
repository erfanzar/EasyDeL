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

"""Async speculative execution: a step may be dispatched before its predecessor commits."""

from __future__ import annotations

import numpy as np
from easydel.inference.esurge.benchmarking import zero_schedule_needs_update
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter

try:
    from ._common import make_core_for_runner, served_request
    from ._common import make_tiny_qwen35 as make_tiny_model
    from .test_mtp_draft_in_verify import _PROMPTS, _make_runner, _serve
except ImportError:  # standalone `python test_x.py`
    from _common import make_core_for_runner, served_request
    from _common import make_tiny_qwen35 as make_tiny_model
    from test_mtp_draft_in_verify import _PROMPTS, _make_runner, _serve


def _serve_async(runner, prompts, *, max_new=10):
    """Drive the runner as the overlap loop does: dispatch ahead whenever the runner allows it.

    Returns the output tokens per request and how many steps were dispatched before their predecessor committed.
    """
    scheduler = make_core_for_runner(
        runner,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        async_scheduling=True,
        num_draft_tokens=runner.num_draft_tokens,
    )
    runner.async_scheduling = True
    reqs = []
    for i, prompt in enumerate(prompts):
        request = served_request(
            request_id=f"r-{i}",
            prompt_token_ids=list(prompt),
            sampling_params=SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True),
            eos_token_id=None,
        )
        scheduler.add_request(request)
        reqs.append(request)

    def _commit(so, output):
        if hasattr(output, "get_output"):
            output = output.get_output()
        # The engine core reads these on the leader only: they must already be host arrays.
        for view in (output.spec_emitted, output.spec_drafts):
            assert view is None or isinstance(view, np.ndarray), type(view)
        scheduler.commit(so, output)

    def _run(so):
        # A step the runner may defer is dispatched and left pending; any other step runs and commits now.
        if runner.can_dispatch_next_before_async_drain(so):
            return so, runner.execute_model_async(so)
        _commit(so, runner.execute_model(so))
        return None

    def _run_empty(so):
        if zero_schedule_needs_update(so):
            _commit(so, runner.execute_model(so))

    # Mirrors the engine's overlap loop: the next step is planned before the pending one commits only when the
    # pending step itself was deferrable, and a planned step the runner refuses to defer runs synchronously.
    pending = None
    overlapped = 0
    for _ in range(4000):
        if pending is None and all(request.is_finished() for request in reqs):
            break
        if pending is not None:
            prev, handle = pending
            pending = None
            if runner.can_dispatch_next_before_async_drain(prev):
                nxt = scheduler.schedule()
                if int(nxt.total_num_scheduled_tokens) > 0:
                    ahead = None
                    if runner.can_dispatch_next_before_async_drain(nxt):
                        ahead = (nxt, runner.execute_model_async(nxt))
                        overlapped += 1
                    _commit(prev, handle)
                    pending = ahead if ahead is not None else _run(nxt)
                    continue
                _commit(prev, handle)
                _run_empty(nxt)
                continue
            _commit(prev, handle)
        so = scheduler.schedule()
        if int(so.total_num_scheduled_tokens) == 0:
            _run_empty(so)
            continue
        pending = _run(so)
    else:
        raise AssertionError("generation did not finish")
    tokens = {request.request_id: list(request.output_token_ids) for request in reqs}
    runner.shutdown()
    runner.executor_manager.kv_pages = None
    return tokens, overlapped


def test_recurrent_layers_without_per_token_states_stay_synchronous():
    """GDN layers keep one state per request, so speculative steps never run ahead and tokens match sync."""
    model = make_tiny_model()
    expected = _serve(
        _make_runner(model, n_seqs=3, drafter=Qwen3_5MTPDrafter(model, num_draft_tokens=3, persist_kv=True)),
        _PROMPTS,
    )
    model = make_tiny_model()
    runner = _make_runner(model, n_seqs=3, drafter=Qwen3_5MTPDrafter(model, num_draft_tokens=3, persist_kv=True))
    actual, overlapped = _serve_async(runner, _PROMPTS)
    assert actual == expected
    assert overlapped == 0
