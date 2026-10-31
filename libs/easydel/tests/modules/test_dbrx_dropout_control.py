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

"""Non-computational DBRX dropout dispatch regression; kernels use symbolic values."""

from types import SimpleNamespace

import pytest
from easydel.modules.dbrx import modeling_dbrx
from easydel.modules.dbrx.dbrx_configuration import DbrxAttentionConfig, DbrxConfig


class _SymbolicTensor:
    """Support shape-only QKV manipulation without allocating or computing arrays."""

    shape = (1, 2, 64)

    def __getitem__(self, key):
        return self

    def reshape(self, *shape):
        return self


class _DropoutKeys:
    """Provide fresh symbolic keys without constructing a JAX RNG container."""

    def __init__(self):
        self.consumed = 0

    @property
    def dropout(self):
        self.consumed += 1
        return f"dropout-key-{self.consumed}"


@pytest.mark.cpu_ok
@pytest.mark.parametrize("probability", [0.0, 0.5])
def test_dbrx_attention_dispatch_owns_and_advances_dropout_rng(monkeypatch, probability):
    config = DbrxConfig(
        d_model=64, n_heads=4, attn_config=DbrxAttentionConfig(kv_n_heads=2, attn_pdrop=probability, clip_qkv=None)
    )
    rngs = _DropoutKeys()

    def symbolic_attention(**kwargs):
        # The symbolic kernel exposes the requested execution semantics through
        # its public output, so a missing RNG or an eval-time advance is visible.
        return SimpleNamespace(attention_outputs=(kwargs["deterministic"], kwargs["dropout_rng"]))

    def define_network(self, config, dtype, param_dtype, precision, rngs):
        self.Wqkv = lambda x: x
        self.out_proj = lambda x: x
        self.attention_performer = SimpleNamespace(forward=symbolic_attention)

    cls = modeling_dbrx.DbrxAttention
    monkeypatch.setattr(cls, "define_network", define_network)
    monkeypatch.setattr(modeling_dbrx, "checkpoint_name", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(cls, "_postprocess_qkv", lambda self, q, k, v: (q, k, v))
    monkeypatch.setattr(cls, "apply_qkv_shardings", lambda self, q, k, v: (q, k, v))
    monkeypatch.setattr(cls, "_apply_rotary", lambda self, q, k, *args: (q, k))
    monkeypatch.setattr(cls, "_merge_heads", lambda self, x: x)
    monkeypatch.setattr(cls, "shard_attention_prod", lambda self, x: x)
    monkeypatch.setattr(
        cls,
        "concatenate",
        lambda self, **kwargs: (
            kwargs["key"],
            kwargs["value"],
            kwargs["mask_info"],
            None,
            kwargs["cache_view"],
            kwargs["cache_metadata"],
        ),
    )
    attention = cls(config, rngs=rngs, layer_idx=0)
    hidden = _SymbolicTensor()
    attention.train()
    first = attention(hidden_states=hidden, mask_info=None, position_ids=None, mode="train").attention_output
    second = attention(hidden_states=hidden, mask_info=None, position_ids=None, mode="train").attention_output
    attention.eval()
    evaluation = attention(hidden_states=hidden, mask_info=None, position_ids=None, mode="train").attention_output

    assert first == (False, "dropout-key-1" if probability else None)
    assert second == (False, "dropout-key-2" if probability else None)
    assert evaluation == (True, None)
    assert rngs.consumed == (2 if probability else 0)
