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

"""Array-free Gemma4 public attention output contracts."""

from types import SimpleNamespace

import pytest
from easydel.modules.gemma4.modeling_gemma4 import Gemma4Attention
from easydel.operations import AttentionOutput

pytestmark = pytest.mark.cpu_ok


class _HostTensor:
    shape = (1, 2, 16)

    def reshape(self, *shape):
        return self


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_explicit_gemma4_attention_request_returns_operation_weights(shared, causal, monkeypatch):
    monkeypatch.setattr("easydel.modules.gemma4.modeling_gemma4.checkpoint_name", lambda value, *args, **kwargs: value)
    attention = Gemma4Attention.__new__(Gemma4Attention)
    hidden, keys, values, output = (_HostTensor() for _ in range(4))
    # Host-only stand-ins for different public attention-weight visibility.
    weights = {False: _HostTensor(), True: _HostTensor()}

    def concatenate(*, causal=True, **kwargs):
        # A cache that bakes visibility into its mask, as transformer caches do.
        mask = SimpleNamespace(causal_mask_baked_in=causal, visibility=weights[causal])
        return keys, values, mask, None, None, None

    def performer(**kwargs):
        # A host backend that honors the public request independently of the
        # model's config, which intentionally disables attention output.
        mask = kwargs.get("mask_info")
        visibility = weights[kwargs["causal"]] if mask is None else mask.visibility
        return AttentionOutput(
            attention_outputs=output,
            attention_weights=visibility if kwargs.get("output_attentions", False) else None,
        )

    attributes = {
        "config": SimpleNamespace(output_attentions=False),
        "num_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "causal": causal,
        "precision": None,
        "is_kv_shared_layer": shared,
        "projection_mapping": {"query_projection": "q_proj", "output_projection": "o_proj"},
        "q_proj": lambda states: hidden,
        "q_norm": lambda states: states,
        "_project_qkv_states": lambda states: (hidden, keys, values),
        "_postprocess_qkv": lambda q, k, v: (q, k, v),
        "apply_qkv_shardings": lambda q, k, v: (q, k, v),
        "_apply_rotary": lambda q, k, *args: (q, k),
        "_kernel_sliding_window": lambda: None,
        "_softmax_aux": lambda: None,
        "concatenate": concatenate,
        "attention_performer": SimpleNamespace(forward=performer),
        "_merge_heads": lambda states: states,
        "shard_attention_prod": lambda states: states,
        "o_proj": lambda states: states,
    }
    for name, value in attributes.items():
        object.__setattr__(attention, name, value)
    result = attention(
        hidden,
        mask_info=None,
        position_ids=None,
        mode=None,
        output_attentions=True,
        shared_key_value=(keys, values) if shared else None,
    )
    assert result.attention_output is output
    assert result.attention_weight is weights[causal]
    assert result.cache_view is None
