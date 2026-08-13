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

"""``config.scan_layers`` must actually lower repeated-layer stacks to lax.scan.

Regression: ~40 modeling files hard-coded ``trace=True`` into
``self.layers.scan`` (spectrax semantics: ``trace=True`` = python unroll), so
``config.scan_layers`` was a silent no-op for those families — flipping it
produced byte-identical compiled programs. The fix ports the
``_layer_scan_trace`` gating pattern (see ``modeling_llama.py`` /
``modeling_qwen3_moe.py``) across the zoo.

This file covers representative fixed families, one per structural class:

* GPT-2 — dense decoder with KV-cache plumbing (``self.h`` stack).
* DBRX — MoE decoder with per-layer router-logit aggregation
  (``self.blocks`` stack).
* RoBERTa — bidirectional encoder with optional cross-attention.
* Whisper — speech seq2seq (encoder and decoder stacks in one forward).

Each family asserts numerical parity between the unrolled and scanned builds
plus a lowering check on the jaxpr: the scanned program must carry more
``scan`` primitives and fewer top-level ``dot_general`` eqns (the stacks below
are structurally homogeneous, so spectrax consolidates them into true
multi-layer scan segments instead of per-layer singletons).
"""

import jax
import numpy as np
import pytest
import spectrax as spx
from jax import numpy as jnp
from jax._src import core as jcore


def _count_prims(jaxpr, counts):
    for eqn in jaxpr.eqns:
        counts[eqn.primitive.name] = counts.get(eqn.primitive.name, 0) + 1
        for v in eqn.params.values():
            for x in v if isinstance(v, (list, tuple)) else [v]:
                inner = getattr(x, "jaxpr", x if isinstance(x, jcore.Jaxpr) else None)
                if inner is not None:
                    _count_prims(inner, counts)


def _assert_scan_lowering_and_parity(build, fwd, atol=1e-5):
    """Build unrolled/scanned twins, assert output parity + scan lowering."""
    m_unrolled = build(False)
    m_scanned = build(True)

    out_unrolled = np.asarray(fwd(m_unrolled))
    out_scanned = np.asarray(fwd(m_scanned))
    np.testing.assert_allclose(out_unrolled, out_scanned, atol=atol, rtol=1e-5)

    counts_unrolled: dict[str, int] = {}
    counts_scanned: dict[str, int] = {}
    _count_prims(jax.make_jaxpr(lambda: fwd(m_unrolled))().jaxpr, counts_unrolled)
    _count_prims(jax.make_jaxpr(lambda: fwd(m_scanned))().jaxpr, counts_scanned)

    # the scanned program carries more scan primitives and fewer unrolled dots
    assert counts_scanned.get("scan", 0) > counts_unrolled.get("scan", 0)
    assert counts_scanned.get("dot_general", 0) < counts_unrolled.get("dot_general", 0)


def _ids(seed: int, vocab: int = 48, shape=(1, 8)):
    return jnp.asarray(np.random.default_rng(seed).integers(0, vocab, shape), np.int32)


def _build_gpt2(scan_layers: bool):
    from easydel.modules.gpt2 import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=48,
        n_positions=64,
        n_embd=32,
        n_layer=4,
        n_head=2,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        scan_layers=scan_layers,
    )
    return GPT2LMHeadModel(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def test_gpt2_dense_scan_layers():
    ids = _ids(0)
    _assert_scan_lowering_and_parity(_build_gpt2, lambda m: m(input_ids=ids).last_hidden_state)


def _build_dbrx(scan_layers: bool):
    from easydel.modules.dbrx import DbrxAttentionConfig, DbrxConfig, DbrxFFNConfig, DbrxForCausalLM

    cfg = DbrxConfig(
        d_model=32,
        n_heads=2,
        n_layers=4,
        max_seq_len=64,
        vocab_size=48,
        attn_config=DbrxAttentionConfig(kv_n_heads=1),
        ffn_config=DbrxFFNConfig(ffn_hidden_size=32, moe_num_experts=4, moe_top_k=2),
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        scan_layers=scan_layers,
    )
    return DbrxForCausalLM(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def test_dbrx_moe_scan_layers():
    ids = _ids(1)
    _assert_scan_lowering_and_parity(_build_dbrx, lambda m: m(input_ids=ids).last_hidden_state)


def test_dbrx_router_logits_force_python_path():
    # growing per-layer router-logit tuples cannot ride a lax.scan carry;
    # requesting them must transparently fall back to the unrolled path
    ids = _ids(2)
    m_scanned = _build_dbrx(True)
    out = m_scanned(input_ids=ids, output_router_logits=True)
    assert out.router_logits is not None
    assert len(out.router_logits) == 4


def _build_roberta(scan_layers: bool):
    from easydel.modules.roberta import RobertaConfig, RobertaModel

    cfg = RobertaConfig(
        vocab_size=48,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        scan_layers=scan_layers,
    )
    return RobertaModel(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def test_roberta_encoder_scan_layers():
    ids = _ids(3)
    _assert_scan_lowering_and_parity(_build_roberta, lambda m: m(input_ids=ids).last_hidden_state)


def _build_whisper(scan_layers: bool):
    from easydel.modules.whisper import WhisperConfig, WhisperForConditionalGeneration

    cfg = WhisperConfig(
        vocab_size=48,
        num_mel_bins=32,
        encoder_layers=2,
        encoder_attention_heads=2,
        decoder_layers=4,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        d_model=32,
        max_source_positions=8,
        max_target_positions=32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        scan_layers=scan_layers,
    )
    return WhisperForConditionalGeneration(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def test_whisper_seq2seq_scan_layers():
    feats = jnp.asarray(np.random.default_rng(4).normal(size=(1, 32, 16)), jnp.float32)
    dec_ids = _ids(5)
    # fp32 reassociation noise through the randomly-initialised decoder can
    # reach ~1e-4 on unlucky inputs (verified exact under JAX_ENABLE_X64:
    # max|d| ~ 4e-16), so whisper uses a looser-than-default tolerance.
    _assert_scan_lowering_and_parity(
        _build_whisper,
        lambda m: m(input_features=feats, decoder_input_ids=dec_ids).logits,
        atol=5e-4,
    )


def test_gpt2_output_hidden_states_force_python_path():
    # per-layer hidden-state tuples cannot ride a lax.scan carry; requesting
    # them must transparently fall back to the unrolled path
    ids = _ids(6)
    m_scanned = _build_gpt2(True)
    out = m_scanned(input_ids=ids, output_hidden_states=True)
    assert out.hidden_states is not None
    assert len(out.hidden_states) >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
