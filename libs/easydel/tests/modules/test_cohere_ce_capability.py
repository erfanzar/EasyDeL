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
"""Cohere CE-only capability and forward contract, without array computation."""

from types import SimpleNamespace

import pytest
from easydel.modules.cohere.modeling_cohere import CohereForCausalLM

pytestmark = pytest.mark.cpu_ok


def test_cohere_hidden_state_forward_does_not_add_an_auxiliary_objective():
    hidden = object()
    cache = object()
    backbone_output = SimpleNamespace(
        last_hidden_state=hidden,
        hidden_states=(hidden,),
        attentions=None,
        past_key_values=cache,
    )
    # Only exercise Python forward orchestration: no parameter initialization,
    # model kernels, projections, or JAX/NumPy arrays are needed for this contract.
    probe = SimpleNamespace(model=lambda **kwargs: backbone_output, config=SimpleNamespace(return_dict=True))

    output = CohereForCausalLM.forward(probe, apply_lm_head=False)

    assert output.logits is None
    assert output.loss is None
    assert getattr(output, "aux_loss", None) is None
    assert output.last_hidden_state is hidden
    assert output.hidden_states == (hidden,)
    assert output.past_key_values is cache
