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

"""Sequence-packing equivalence tests for Qwen3.5.

Qwen3.5 is a hybrid model: gated-delta-rule (GDR) linear-attention layers + full
self-attention layers. When several documents are packed into one sequence (with a
per-token ``segment_ids`` array), packed inference must be identical to running each
document on its own — i.e. documents must NOT attend to, nor carry recurrent state
from, one another:

* full attention -> block-diagonal mask per document;
* GDR linear attention -> recurrence (and the depthwise causal conv) reset at each
  document boundary.

Each test packs ``[docA, docB]`` and asserts the packed per-token logits match the
separately-computed unpacked logits for both documents.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

ATOL = 2e-3


def _build(layer_types):
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from easydel.modules.qwen3_5.qwen3_5_configuration import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        layer_types=layer_types,
        mtp_num_hidden_layers=0,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
        scan_layers=False,
    )
    return Qwen3_5ForCausalLM(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _logits(model, ids):
    return np.asarray(model(input_ids=jnp.asarray(ids)).logits.astype(jnp.float32))[0]


def _packed_logits(model, packed, seg):
    # Sequence packing is carried via ``mask_info`` (the data pipeline emits ``segment_ids``,
    # which the trainer's ``compute_loss`` folds into ``mask_info`` once). Mirror that here.
    from ejkernel.types import MaskInfo

    mask_info = MaskInfo.from_segments(q_segment_ids=jnp.asarray(seg, dtype=jnp.int32))
    out = model(input_ids=jnp.asarray(packed), mask_info=mask_info)
    return np.asarray(out.logits.astype(jnp.float32))[0]


@pytest.mark.parametrize(
    "layer_types",
    [
        pytest.param(["full_attention", "full_attention"], id="full_attention"),
        pytest.param(["linear_attention", "full_attention"], id="hybrid_gdr_full"),
        pytest.param(["linear_attention", "linear_attention"], id="linear_only"),
    ],
)
def test_packed_equals_unpacked(layer_types):
    """Packed [docA, docB] per-token logits must equal the unpacked per-doc logits."""
    model = _build(layer_types)

    doc_a = np.array([[5, 9, 2, 7, 1]], dtype="int32")  # len 5
    doc_b = np.array([[3, 8, 4]], dtype="int32")  # len 3
    la = _logits(model, doc_a)
    lb = _logits(model, doc_b)

    packed = np.concatenate([doc_a, doc_b], axis=1)
    seg = np.array([[0, 0, 0, 0, 0, 1, 1, 1]], dtype="int32")
    lp = _packed_logits(model, packed, seg)

    da = float(np.max(np.abs(lp[:5] - la)))
    db = float(np.max(np.abs(lp[5:] - lb)))
    assert da < ATOL, f"{layer_types}: docA contaminated by packing, max|Δ|={da:.2e}"
    assert db < ATOL, f"{layer_types}: docB contaminated by previous doc, max|Δ|={db:.2e}"


def test_gdr_packed_gradients_match_unpacked():
    """GDR linear-attention BACKWARD must also be segment-aware: gradients from a packed
    [docA, docB] forward must equal the sum of per-document gradients. Guards against the
    fwd-only trap where a segment-blind custom_vjp silently corrupts packed gradients.
    """
    import jax
    from ejkernel.kernels._xla.gated_delta_rule._xla_impl_fwd import _chunk_gdr_fwd

    b, h, k, v, la, lb = 1, 2, 4, 4, 5, 3
    rng = np.random.default_rng(0)

    def rnd(*s):
        return jnp.asarray(rng.standard_normal(s), dtype=jnp.float32)

    q, kk, vv = rnd(b, h, la + lb, k), rnd(b, h, la + lb, k), rnd(b, h, la + lb, v)
    beta = jax.nn.sigmoid(rnd(b, h, la + lb))
    decay = -jax.nn.softplus(rnd(b, h, la + lb)) * 0.5
    seg = jnp.array([[0] * la + [1] * lb], dtype=jnp.int32)

    def loss_packed(q, kk, vv, beta, decay):
        o, _ = _chunk_gdr_fwd(q, kk, vv, beta, decay, seg_ids=seg, chunk_size=64)
        return jnp.sum(o**2)

    def loss_docs(q, kk, vv, beta, decay):
        oa, _ = _chunk_gdr_fwd(
            q[:, :, :la], kk[:, :, :la], vv[:, :, :la], beta[:, :, :la], decay[:, :, :la], chunk_size=64
        )
        ob, _ = _chunk_gdr_fwd(
            q[:, :, la:], kk[:, :, la:], vv[:, :, la:], beta[:, :, la:], decay[:, :, la:], chunk_size=64
        )
        return jnp.sum(oa**2) + jnp.sum(ob**2)

    gp = jax.grad(loss_packed, argnums=(0, 1, 2, 3, 4))(q, kk, vv, beta, decay)
    gd = jax.grad(loss_docs, argnums=(0, 1, 2, 3, 4))(q, kk, vv, beta, decay)
    for name, a, c in zip(["dq", "dk", "dv", "dbeta", "ddecay"], gp, gd, strict=True):
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(c))))
        assert d < 1e-4, f"GDR packed gradient {name} diverges from per-doc, max|Δ|={d:.2e}"


def test_compute_loss_converts_segment_ids():
    """``compute_loss`` (the trainer entry) folds ``segment_ids`` into ``mask_info`` and
    isolates packed documents — the path SFT/Distillation actually use."""
    model = _build(["linear_attention", "full_attention"])
    packed = np.array([[5, 9, 2, 7, 1, 3, 8, 4]], dtype="int32")
    seg = np.array([[0, 0, 0, 0, 0, 1, 1, 1]], dtype="int32")
    am = np.ones_like(packed)
    out, _metrics = model.compute_loss(
        input_ids=jnp.asarray(packed),
        attention_mask=jnp.asarray(am),
        segment_ids=jnp.asarray(seg),
        labels=jnp.asarray(packed),
    )
    assert np.isfinite(float(out.loss))


def test_direct_model_call_converts_segment_ids():
    """All EasyDeL modules inherit the direct-call segment_ids folding path."""
    model = _build(["full_attention"])
    packed = np.array([[5, 9, 2, 7, 1, 3, 8, 4]], dtype="int32")
    seg = np.array([[0, 0, 0, 0, 0, 1, 1, 1]], dtype="int32")

    out = model(input_ids=jnp.asarray(packed), segment_ids=jnp.asarray(seg))
    logits = np.asarray(out.logits.astype(jnp.float32))

    assert logits.shape == (1, 8, 128)
    assert np.all(np.isfinite(logits))


def test_vanilla_packed_attention_does_not_materialize_mask_info(monkeypatch):
    """Vanilla must consume packed segment ids directly instead of expanding MaskInfo's
    dense attention mask first.
    """
    from ejkernel.types import MaskInfo

    def fail_materialize(*args, **kwargs):
        del args, kwargs
        raise AssertionError("vanilla packed attention materialized MaskInfo attention_mask")

    monkeypatch.setattr(MaskInfo, "get_or_compute_attention_mask", fail_materialize)

    model = _build(["full_attention"])
    packed = np.array([[5, 9, 2, 7, 1, 3, 8, 4]], dtype="int32")
    seg = np.array([[0, 0, 0, 0, 0, 1, 1, 1]], dtype="int32")
    logits = _packed_logits(model, packed, seg)

    assert logits.shape == (8, 128)
    assert np.all(np.isfinite(logits))


def test_qwen3_next_segment_ids_extend_to_internal_hidden_length():
    from easydel.modules.qwen3_next.modeling_qwen3_next import _normalize_packed_segment_ids

    seg = jnp.array([[0, 0, 0, 1, 1, 2, 2, 2]], dtype=jnp.int32)

    normalized = _normalize_packed_segment_ids(seg, 9)

    assert normalized.shape == (1, 9)
    assert normalized.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2]]


def test_unpacked_forward_unchanged():
    """A forward without ``segment_ids`` must run on the original (non-packing) path."""
    model = _build(["linear_attention", "full_attention"])
    ids = np.random.default_rng(0).integers(0, 128, size=(2, 16)).astype("int32")
    logits = np.asarray(model(input_ids=jnp.asarray(ids)).logits.astype(jnp.float32))
    assert logits.shape == (2, 16, 128)
    assert np.all(np.isfinite(logits))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
