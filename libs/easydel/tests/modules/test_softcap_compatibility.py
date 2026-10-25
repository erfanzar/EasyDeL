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

"""Public soft-cap compatibility and dispatch contracts without array computation."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import SoftCappedAttentionModule, _flexible
from easydel.modules.gemma2 import modeling_gemma2
from easydel.modules.gemma3 import modeling_gemma3
from easydel.operations import AttentionOutput

pytestmark = pytest.mark.cpu_ok


def test_legacy_softcap_names_resolve_to_shared_public_class():
    assert modeling_gemma2.Gemma2SoftCappedAttentionModule is SoftCappedAttentionModule
    assert modeling_gemma3.Gemma2SoftCappedAttentionModule is SoftCappedAttentionModule
    assert modeling_gemma3.SoftCappedAttentionModule is SoftCappedAttentionModule


class _CapOutput:
    def __init__(self, cap):
        self.cap = cap

    def astype(self, dtype):
        return self


class _CapOperation:
    metadata = SimpleNamespace(runtime_dtype=None)

    def get_impl_name(self):
        return "host_contract_backend"

    def __call__(self, *, logits_soft_cap, **kwargs):
        return AttentionOutput(attention_outputs=_CapOutput(logits_soft_cap))


@pytest.mark.parametrize(
    "performer_class",
    [modeling_gemma2.Gemma2SoftCappedAttentionModule, modeling_gemma3.Gemma2SoftCappedAttentionModule],
    ids=["gemma2-legacy", "gemma3-legacy"],
)
@pytest.mark.parametrize("default_cap", [None, 50.0])
@pytest.mark.parametrize("call_cap", [None, 30.0, 0.0])
@pytest.mark.parametrize("method", ["forward", "__call__"])
def test_legacy_softcap_default_and_override_reach_dispatched_output(
    monkeypatch, performer_class, default_cap, call_cap, method
):
    # The backend emits the supplied cap as a host-only output, exercising public dispatch
    # without running attention numerics or allocating JAX arrays.
    monkeypatch.setattr(_flexible, "_attention_mesh_context", lambda config: nullcontext())
    config = EasyDeLBaseConfig(attn_mechanism="vanilla", decode_attn_mechanism=None)
    module = performer_class(config, softmax_scale=1.0, logits_soft_cap=default_cap, requires_cache=False)
    object.__setattr__(module, "impl", _CapOperation())
    output = getattr(module, method)(None, None, None, mode=None, logits_soft_cap=call_cap)
    assert output.attention_outputs.cap == (default_cap if call_cap is None else call_cap)
    assert module.logits_soft_cap == default_cap
