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

"""DeepSeek-V4's pure-decode fast path must equal the general packed path.

``_packed_forward`` scatters the flat eSurge token stream into a dense
``[num_slots, total_tokens, ...]`` grid and scans its columns. In a decode
bucket every request contributes exactly one token, so only column 0 is ever
occupied and that grid is quadratic waste: profiling a cc128 decode step
measured 59.8% of device time in data movement, `broadcast` alone at 13.8 ms
(the grid's ``jnp.zeros`` is 1.07 GB per layer), against 3.6% for the expert
matmul. The fast path collapses the token axis to ``[num_slots, ...]`` and
drops the column loop.

It is selected by ``inference_mode_forces_decode()``, so the risk it carries is
a *silent* one: if the two paths ever disagree, generation quietly corrupts
rather than failing. These tests pin them together.
"""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from easydel.infra.sharding import decode_mode_specs
from jax import numpy as jnp

SLOTS = 4


def _model(num_layers: int = 4):
    import easydel as ed
    from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config

    config = DeepseekV4Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        max_position_embeddings=256,
        sliding_window=16,
        index_topk=8,
        index_n_heads=4,
        index_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        head_dim=32,
        o_groups=2,
    )
    # Pin a single-device mesh. The default fills ep=-1 with every visible
    # device, and under the 8-fake-device CPU trio that is ep=8 against this
    # model's 4 routed experts -- the fused-MoE shard_map then rejects the
    # expert axis as indivisible. This comparison is about the decode path,
    # not sharding, so an explicit 1-device mesh keeps it device-count stable.
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    return ed.AutoEasyDeLModelForCausalLM.from_config(config, dtype=jnp.float32, param_dtype=jnp.float32)


class _PackedMetadata:
    """Minimal stand-in for eSurge's cache metadata (duck-typed by the model).

    One token per request is exactly the decode invariant the executor derives
    its bucket layout from (``num_tokens == padded_num_reqs``).
    """

    def __init__(self, num_slots: int):
        self.query_start_loc = jnp.arange(num_slots + 1, dtype=jnp.int32)
        self.num_seqs = jnp.asarray([num_slots], dtype=jnp.int32)
        self.recurrent_state_indices = None


def _run(model, cache, *, decode: bool, positions: int = 5):
    """One packed step through either path, returning logits and cache state."""
    ids = jnp.asarray([[3, 9, 5, 1][:SLOTS]], dtype=jnp.int32)
    pos = jnp.full((1, SLOTS), positions, dtype=jnp.int32)
    with decode_mode_specs(decode):
        out = model(
            input_ids=ids,
            position_ids=pos,
            past_key_values=cache,
            cache_metadata=_PackedMetadata(SLOTS),
        )
    return np.asarray(out.logits, np.float32), out.past_key_values


def _cache(model):
    return model.init_cache(batch_size=SLOTS, max_length=64)


def test_decode_fastpath_matches_general_path_logits():
    """The whole point: identical outputs, or generation silently corrupts."""
    model = _model()
    general, _ = _run(model, _cache(model), decode=False)
    fast, _ = _run(model, _cache(model), decode=True)

    assert general.shape == fast.shape == (1, SLOTS, 128)
    assert np.all(np.isfinite(fast)), "fast path produced non-finite logits"
    delta = float(np.max(np.abs(general - fast)))
    assert delta < 1e-4, f"decode fast path diverged from the packed path, max|delta|={delta:.3e}"


def test_decode_fastpath_leaves_identical_cache_state():
    """Divergence in the cache would corrupt the NEXT step, not this one."""
    model = _model()
    _, cache_general = _run(model, _cache(model), decode=False)
    _, cache_fast = _run(model, _cache(model), decode=True)

    checked = 0
    for gview, fview in zip(cache_general.views, cache_fast.views, strict=True):
        if gview is None or fview is None:
            continue
        for name, gval in vars(gview).items():
            fval = getattr(fview, name, None)
            if not hasattr(gval, "shape") or not hasattr(fval, "shape"):
                continue
            g = np.asarray(gval, np.float32)
            f = np.asarray(fval, np.float32)
            assert g.shape == f.shape, f"{name}: shape {g.shape} vs {f.shape}"
            # -inf gates are meaningful state, so treat matching -inf as equal
            # rather than as the NaN that -inf minus -inf would produce. The
            # subtraction is masked BEFORE it happens; np.where would evaluate
            # both branches and warn on the discarded NaN.
            both_inf = np.isneginf(g) & np.isneginf(f)
            gm = np.where(both_inf, 0.0, g)
            fm = np.where(both_inf, 0.0, f)
            delta = np.max(np.abs(gm - fm)) if g.size else 0.0
            assert delta < 1e-4, f"cache leaf {name} diverged, max|delta|={float(delta):.3e}"
            checked += 1
    assert checked > 0, "no cache leaves were compared -- the test proved nothing"


@pytest.mark.parametrize("position", [0, 5, 33])
def test_decode_fastpath_matches_across_stream_positions(position):
    """Position 0 exercises the prefill-from-empty self-heal branch."""
    model = _model(num_layers=2)
    general, _ = _run(model, _cache(model), decode=False, positions=position)
    fast, _ = _run(model, _cache(model), decode=True, positions=position)
    delta = float(np.max(np.abs(general - fast)))
    assert delta < 1e-4, f"paths diverged at position {position}, max|delta|={delta:.3e}"


def test_fastpath_actually_fires_and_matches_the_grid_path():
    """Pin the collapse itself, not merely that two runs agree.

    The other tests compare ``decode=True`` against ``decode=False``. That is
    meaningful only while the collapse exists: delete it and both sides take
    the ``[num_slots, T, ...]`` grid, so they agree trivially and keep passing
    with the optimization gone. This test asserts the gate is open and then
    forces the general path back on inside the same scope, so the two sides
    are genuinely different code.
    """
    import easydel.modules.deepseek_v4.modeling_deepseek_v4 as mod

    assert not mod.inference_mode_forces_decode(), "decode gate should be closed outside the scope"
    with decode_mode_specs(True):
        assert mod.inference_mode_forces_decode(), "decode gate did not open; test proves nothing"

    model = _model(num_layers=2)
    fast, _ = _run(model, _cache(model), decode=True)
    original = mod.inference_mode_forces_decode
    try:
        mod.inference_mode_forces_decode = lambda *_a, **_k: False
        grid, _ = _run(model, _cache(model), decode=True)
    finally:
        mod.inference_mode_forces_decode = original

    delta = float(np.max(np.abs(fast - grid)))
    assert delta == 0.0, f"collapsed decode path is not the grid path, max|delta|={delta:.3e}"
