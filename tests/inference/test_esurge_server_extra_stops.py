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

from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.sampling_params import SamplingParams


def _make_server(extra_stops):
    server = object.__new__(eSurgeApiServer)
    server._extra_stops = eSurgeApiServer._normalize_stop_sequences(extra_stops)
    return server


def test_normalize_stop_sequences_handles_common_inputs():
    assert eSurgeApiServer._normalize_stop_sequences(None) == []
    assert eSurgeApiServer._normalize_stop_sequences("<user>") == ["<user>"]
    assert eSurgeApiServer._normalize_stop_sequences(["", "<user>", "<user>", 42, None]) == ["<user>", "42"]


def test_apply_extra_stops_appends_and_deduplicates():
    server = _make_server(["<user>", "</assistant>"])
    sampling_params = SamplingParams(max_tokens=32, stop=["</assistant>", "DONE"])

    updated = server._apply_extra_stops_to_sampling_params(sampling_params)

    assert updated.stop == ["</assistant>", "DONE", "<user>"]


def test_apply_extra_stops_populates_empty_stop_list():
    server = _make_server("<user>")
    sampling_params = SamplingParams(max_tokens=16)

    updated = server._apply_extra_stops_to_sampling_params(sampling_params)

    assert updated.stop == ["<user>"]
