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

"""Request-plane teardown remains observable to pending callers with logging enabled."""

import logging
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed import wire
from easydel.inference.esurge.distributed.request_plane import OriginRequestPlane, RequestPlaneError
from easydel.inference.sampling_params import SamplingParams


@pytest.mark.cpu_ok
def test_fail_pending_rejects_admission_and_retires_remote_ownership():
    class Coordinator:
        def set_plane_handler(self, handler):
            self.handler = handler

        def send_to_leader(self, message, payload=None):
            if isinstance(message, wire.AbortReq):
                return
            assert isinstance(message, wire.Admit)
            # A teardown races with the admission's acknowledgement.
            plane.fail_pending("owner disconnected")

    plane = OriginRequestPlane(
        coordinator=Coordinator(),
        submit_outputs=lambda bundle: None,
        alive=lambda: True,
        rank=1,
        admit_timeout_s=0.01,
        info=logging.getLogger(__name__).info,
    )
    request = SimpleNamespace(
        request_id="pending",
        parent_request_id=None,
        sample_index=0,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
    )

    with pytest.raises(RequestPlaneError, match="owner disconnected"):
        plane.submit_remote([request])

    assert not plane.is_remote("pending")
    assert not plane.notify_abort("pending")
