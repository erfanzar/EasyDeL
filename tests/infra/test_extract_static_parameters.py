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

import inspect

from easydel.infra.utils import extract_static_parameters


def test_extract_static_parameters_handles_wrapped_cycle():
    def call(self, hidden_states, mode, frequencies=None, output_attentions=False):
        return hidden_states

    # Simulate a broken decorator chain that inspect.unwrap() cannot traverse.
    call.__wrapped__ = call

    class DummyModule:
        __call__ = call

    try:
        inspect.signature(DummyModule.__call__)
    except ValueError as err:
        assert "wrapper loop when unwrapping" in str(err)

    assert extract_static_parameters(DummyModule) == (2, 3, 4)
