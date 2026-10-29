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

"""Regression test for eSurge.generate() output ordering (bug E1).

``generate`` used to append finished outputs in *completion* order, so a batch
where a later prompt finishes first came back re-ordered. Positional callers
(``for i, out in enumerate(outputs)``, benchmarks, RLHF rollouts) then paired
the wrong prompt with the wrong text. The engine must return outputs aligned
to the input ``prompts`` / ``request_ids`` order regardless of finish order.
"""

from __future__ import annotations

import threading

from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput, eSurge
from easydel.inference.sampling_params import SamplingParams


class _OutOfOrderEngine(eSurge):
    """eSurge stub that finishes the *second* request before the first.

    Only the machinery ``generate`` touches is stubbed; no model/scheduler is
    built. ``_raise_if_scheduler_failed`` is (ab)used as the per-iteration hook
    that flips ``finished`` on the pre-registered outputs so that request "B"
    (the second prompt) completes on step 1 and "A" (the first) on step 2.
    """

    def __init__(self, request_ids: list[str]):
        self._output_event = threading.Event()
        self._output_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._request_events: dict[str, object] = {}
        self._max_request_outputs = None
        self._scheduler_running = True  # drives the _generation_alive property
        self._step = 0
        self._finish_order = list(reversed(request_ids))  # later prompt finishes first
        self._request_outputs = {
            rid: RequestOutput(
                request_id=rid,
                prompt=f"prompt-{rid}",
                prompt_token_ids=[1, 2, 3],
                outputs=[CompletionOutput(index=0, text=f"text-{rid}", token_ids=[9])],
                finished=False,
                accumulated_text=f"text-{rid}",
            )
            for rid in request_ids
        }

    # --- stubs for the periphery generate() calls -------------------------
    def _generate_request_id(self) -> str:  # pragma: no cover - not used (ids supplied)
        return "auto"

    def _tokenize_prompt(self, request_id, prompt):
        return [1, 2, 3]

    def _prepare_sampling_params_for_request(self, base, *, request_id, prompt):
        return base

    def _add_request(self, *args, **kwargs):
        return None

    def _submit_scheduler_requests(self, requests):  # pragma: no cover - never called
        return None

    def _ensure_scheduler_running(self, context=None):
        return None

    def _recover_orphaned_request(self, request_id):
        return None

    def _raise_if_scheduler_failed(self):
        # Advance the simulated timeline: reveal one more finished output per
        # iteration, in reverse-of-prompt order.
        if self._step < len(self._finish_order):
            self._request_outputs[self._finish_order[self._step]].finished = True
        self._step += 1
        self._output_event.set()


def test_generate_returns_outputs_in_prompt_order_not_finish_order():
    request_ids = ["A", "B"]
    engine = _OutOfOrderEngine(request_ids)

    outputs = engine.generate(
        ["long prompt that finishes last", "short"],
        SamplingParams(max_tokens=4),
        request_id=request_ids,
        use_tqdm=False,
    )

    # Even though "B" finished before "A", the returned list is prompt-aligned.
    assert [o.request_id for o in outputs] == ["A", "B"]
    assert outputs[0].outputs[0].text == "text-A"
    assert outputs[1].outputs[0].text == "text-B"
