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

"""Drafter semantics the eSurge speculative strategy must match.

Covers the Gemma4 assistant against the HF reference
(``SinglePositionMultiTokenCandidateGenerator`` / ``Gemma4AssistantForCausalLM``):

* target-K/V layer mapping reuses the target's last NON-KV-shared layer of the
  same attention type (E2B/E4B shared layers own no cache view);
* Q-only attention uses ``scaling=1.0`` like every Gemma4 attention;
* sliding layers only see the last ``sliding_window + 1`` target positions;
* every draft step of a window queries ONE constant position ``L - 1``;

and the block-drafter (DSpark / DFlash) loop contract: fixed anchor + target
context, drafted tokens appended, drafter hidden never fed back.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import easydel as ed  # noqa: F401  # ensures registry side effects fire
import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.runners.spec import default_assistant_layer_mapping
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import DraftStep, Gemma4AssistantDrafter
from easydel.modules.gemma4_assistant import (
    Gemma4AssistantConfig,
    Gemma4AssistantForCausalLM,
    Gemma4AssistantTextConfig,
)
from easydel.modules.gemma4_assistant.modeling_gemma4_assistant import Gemma4AssistantQOnlyAttention
from jax import numpy as jnp

_HIDDEN = 64
_BACKBONE = 128
_VOCAB = 256
_WINDOW = 32
_ASSISTANT_TYPES = ["sliding_attention"] * 3 + ["full_attention"]
_REQ_STATE = types.SimpleNamespace(sampling_params=SamplingParams(temperature=0.0, max_tokens=32))


def _make_assistant() -> Gemma4AssistantForCausalLM:
    text_cfg = Gemma4AssistantTextConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_HIDDEN * 4,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=_HIDDEN // 2,
        global_head_dim=_HIDDEN,
        layer_types=list(_ASSISTANT_TYPES),
        sliding_window=_WINDOW,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    config = Gemma4AssistantConfig(
        text_config=text_cfg,
        backbone_hidden_size=_BACKBONE,
        num_centroids=16,
        centroid_intermediate_top_k=4,
        use_ordered_embeddings=True,
        tie_word_embeddings=True,
    )
    return Gemma4AssistantForCausalLM(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _make_drafter(target_config=None) -> Gemma4AssistantDrafter:
    target_embed = spx.nn.Embed(_VOCAB, _BACKBONE, rngs=spx.Rngs(1), dtype=jnp.float32, param_dtype=jnp.float32)
    return Gemma4AssistantDrafter(
        assistant_model=_make_assistant(),
        target_embed_module=target_embed,
        target_config=target_config,
    )


def _gemma4_layer_types(num_layers: int, pattern: int) -> list[str]:
    return ["full_attention" if (i + 1) % pattern == 0 else "sliding_attention" for i in range(num_layers)]


# --------------------------------------------------------------------------- mapping


def test_mapping_reuses_last_non_shared_layer_of_same_type():
    """KV-shared target (E2B-like): map to the donor layers, never to shared ones."""
    target_types = _gemma4_layer_types(35, 5)
    mapping = default_assistant_layer_mapping(
        4,
        35,
        assistant_layer_types=_ASSISTANT_TYPES,
        target_layer_types=target_types,
        num_kv_shared_layers=20,
    )
    # First KV-shared layer is 15: last sliding donor is 13, last full donor is 14.
    assert mapping == [13, 13, 13, 14]
    assert all(idx < 35 - 20 for idx in mapping), "a KV-shared layer owns no cache view"


def test_mapping_without_kv_sharing_uses_last_layer_of_each_type():
    """No KV sharing: every sliding assistant layer reads the LAST sliding target layer."""
    mapping = default_assistant_layer_mapping(
        4,
        60,
        assistant_layer_types=_ASSISTANT_TYPES,
        target_layer_types=_gemma4_layer_types(60, 6),
        num_kv_shared_layers=0,
    )
    assert mapping == [58, 58, 58, 59]


def test_mapping_without_layer_types_keeps_legacy_heuristic():
    assert default_assistant_layer_mapping(4, 35) == [31, 32, 33, 34]
    assert default_assistant_layer_mapping(4, 2) == [1, 1, 1, 1]


def test_drafter_resolves_mapping_from_target_config():
    text_config = types.SimpleNamespace(
        num_hidden_layers=35,
        layer_types=_gemma4_layer_types(35, 5),
        num_kv_shared_layers=20,
    )
    drafter = _make_drafter(target_config=types.SimpleNamespace(text_config=text_config))
    assert drafter.resolve_layer_mapping() == [13, 13, 13, 14]


# --------------------------------------------------------------------------- attention


def test_q_only_attention_uses_unit_softmax_scale():
    """HF Gemma4 attention (and EasyDeL's gemma4) use ``scaling=1.0``."""
    cfg = _make_assistant().config.text_config
    attn = Gemma4AssistantQOnlyAttention(cfg, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(3))
    b, s, kv = 1, 2, 5
    key = jax.random.PRNGKey(4)
    hidden = jax.random.normal(jax.random.fold_in(key, 0), (b, s, _HIDDEN), dtype=jnp.float32)
    k = jax.random.normal(jax.random.fold_in(key, 1), (b, kv, attn.num_heads, attn.head_dim), dtype=jnp.float32)
    v = jax.random.normal(jax.random.fold_in(key, 2), (b, kv, attn.num_heads, attn.head_dim), dtype=jnp.float32)
    pos = jnp.asarray([[7, 8]], dtype=jnp.int32)

    got = attn(hidden, k, v, pos, None)

    q = attn.q_norm(attn.q_proj(hidden).reshape(b, s, attn.num_heads, attn.head_dim))
    q = attn._apply_query_rope(q, pos)
    scores = jnp.einsum("bshd,bthd->bhst", q, k)  # no 1/sqrt(head_dim)
    out = jnp.einsum("bhst,bthd->bshd", jax.nn.softmax(scores, axis=-1), v).reshape(b, s, -1)
    want = attn.o_proj(out)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-5)


def test_target_attention_mask_limits_sliding_layers_to_window():
    drafter = _make_drafter()
    masks = drafter.build_target_attention_mask(kv_len=128, context_len=100)
    full = np.asarray(masks["full_attention"]).reshape(-1) == 0.0
    sliding = np.asarray(masks["sliding_attention"]).reshape(-1) == 0.0
    assert np.flatnonzero(full).tolist() == list(range(100))
    # HF: |q - kv| <= window over the flipped kv axis -> the last window + 1 positions.
    assert np.flatnonzero(sliding).tolist() == list(range(100 - 1 - _WINDOW, 100))


def test_per_layer_type_masks_reach_their_layers(monkeypatch):
    """A dict mask routes the sliding entry to sliding layers and the full one to the full layer."""
    model = _make_assistant()
    seen: list[tuple[str, jax.Array]] = []
    original = Gemma4AssistantQOnlyAttention.forward

    def _spy(self, hidden_states, key_states=None, value_states=None, position_ids=None, attention_mask=None):
        seen.append((self.layer_type, attention_mask))
        return original(self, hidden_states, key_states, value_states, position_ids, attention_mask)

    monkeypatch.setattr(Gemma4AssistantQOnlyAttention, "forward", _spy)
    kv_len = 4
    full = jnp.zeros((1, 1, 1, kv_len), dtype=jnp.float32)
    sliding = jnp.full((1, 1, 1, kv_len), -1.0e10, dtype=jnp.float32).at[..., -1].set(0.0)
    pairs = []
    for layer in model.model.layers:
        kv = jnp.ones((1, kv_len, layer.self_attn.num_heads, layer.self_attn.head_dim), dtype=jnp.float32)
        pairs.append((kv, kv))
    model(
        backbone_hidden_states=jnp.ones((1, 1, _BACKBONE), dtype=jnp.float32),
        target_token_embeds=jnp.ones((1, 1, _BACKBONE), dtype=jnp.float32),
        target_key_value_pairs=pairs,
        attention_mask={"full_attention": full, "sliding_attention": sliding},
    )
    assert [layer_type for layer_type, _ in seen] == _ASSISTANT_TYPES
    for layer_type, mask in seen:
        expected = sliding if layer_type == "sliding_attention" else full
        np.testing.assert_array_equal(np.asarray(mask), np.asarray(expected))


# --------------------------------------------------------------------------- strategy loop


class _RecordingAssistant:
    """Target-K/V drafter stand-in that records every draft call."""

    requires_target_kv_cache = True
    supports_return_full_log_probs = True
    draft_position_offset = 1
    advance_draft_position = False

    def __init__(self):
        self.assistant = types.SimpleNamespace(
            config=types.SimpleNamespace(text_config=types.SimpleNamespace(sliding_window=4))
        )
        self.calls: list[dict] = []

    build_target_attention_mask = Gemma4AssistantDrafter.build_target_attention_mask

    def resolve_layer_mapping(self, target_cache=None):
        del target_cache
        return [0]

    def draft(self, **kwargs):
        self.calls.append(kwargs)
        step = len(self.calls)
        return DraftStep(
            token_ids=jnp.asarray([10 + step], dtype=jnp.int32),
            hidden_states=jnp.full((1, 1, 8), float(step), dtype=jnp.float32),
        )


def _stub_runner(max_model_len: int = 16):
    view = types.SimpleNamespace(
        key=jnp.zeros((1, max_model_len, 1, 4), dtype=jnp.float32),
        value=jnp.zeros((1, max_model_len, 1, 4), dtype=jnp.float32),
    )
    return types.SimpleNamespace(
        max_num_reqs=1,
        max_num_seqs=1,
        requests={},
        max_model_len=max_model_len,
        sequence_buffer=types.SimpleNamespace(req_id_to_index={"r": 0}),
        executor_manager=types.SimpleNamespace(kv_pages=types.SimpleNamespace(views=[view])),
    )


def test_gemma4_assistant_drafter_declares_hf_query_position():
    drafter = _make_drafter()
    assert drafter.draft_position_offset == 1
    assert drafter.advance_draft_position is False


def test_assistant_window_uses_constant_query_position_and_windowed_masks(monkeypatch):
    """Every step queries position ``L - 1`` (= seed_position + 1) with the HF masks."""
    monkeypatch.delenv("EASYDEL_SPEC_ADAPTIVE_CONF", raising=False)
    drafter = _RecordingAssistant()
    strategy = DrafterSpeculation(runner=_stub_runner(), drafter=drafter, num_draft_tokens=3)
    seed_position = 9  # L - 2: the seed hidden's position; the seed token sits at L - 1 = 10
    drafted = strategy.draft_next(
        req_id="r",
        seed_token=5,
        seed_position=seed_position,
        seed_hidden=jnp.zeros((8,), dtype=jnp.float32),
        req_state=_REQ_STATE,
    )
    assert drafted == [11, 12, 13]
    positions = [int(np.asarray(call["position_ids"]).reshape(-1)[0]) for call in drafter.calls]
    assert positions == [seed_position + 1] * 3
    # The drafter's projected hidden is fed back as the next step's target hidden (HF).
    fed = [float(np.asarray(call["target_hidden_states"]).reshape(-1)[0]) for call in drafter.calls]
    assert fed == [0.0, 1.0, 2.0]
    mask = drafter.calls[0]["attention_mask"]
    full = np.flatnonzero(np.asarray(mask["full_attention"]).reshape(-1) == 0.0).tolist()
    sliding = np.flatnonzero(np.asarray(mask["sliding_attention"]).reshape(-1) == 0.0).tolist()
    assert full == list(range(seed_position + 1))
    assert sliding == list(range(seed_position - 4, seed_position + 1))


class _RecordingBlockDrafter:
    """Block drafter stand-in (DSpark / DFlash contract)."""

    block_draft = True
    supports_return_full_log_probs = True

    def __init__(self):
        self.calls: list[dict] = []

    def draft(self, **kwargs):
        self.calls.append(kwargs)
        step = len(self.calls)
        return DraftStep(
            token_ids=jnp.asarray([20 + step], dtype=jnp.int32),
            hidden_states=jnp.full((1, 1, 8), -1.0, dtype=jnp.float32),
        )


def test_block_drafter_gets_anchor_plus_drafts_and_fixed_context(monkeypatch):
    monkeypatch.delenv("EASYDEL_SPEC_ADAPTIVE_CONF", raising=False)
    drafter = _RecordingBlockDrafter()
    runner = types.SimpleNamespace(max_num_reqs=1, max_num_seqs=1, requests={})
    strategy = DrafterSpeculation(runner=runner, drafter=drafter, num_draft_tokens=3)
    seed_hidden = jnp.arange(8, dtype=jnp.float32)
    drafted = strategy.draft_next(
        req_id="r",
        seed_token=5,
        seed_position=9,
        seed_hidden=seed_hidden,
        req_state=_REQ_STATE,
    )
    assert drafted == [21, 22, 23]
    assert [np.asarray(call["input_ids"]).tolist() for call in drafter.calls] == [[[5]], [[5, 21]], [[5, 21, 22]]]
    for call in drafter.calls:
        np.testing.assert_array_equal(np.asarray(call["target_hidden_states"]).reshape(-1), np.asarray(seed_hidden))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
