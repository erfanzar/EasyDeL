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

"""A tied-embedding LM head must project with a kernel in the head's own column-parallel layout.

The embedding is stored over ``(vocab, hidden)`` with hidden on the tensor axis, so its transpose
arrives over ``(hidden, vocab)`` with the contracting dimension tensor-sharded. Unconstrained, GSPMD
may lower the projection as partial full-vocab matmuls plus an all-reduce that materializes
replicated logits. At test scale XLA replicates the small result either way, so the deterministic,
scale-independent observable is the layout of the kernel the head actually multiplies with.
"""

import easydel as ed
import jax
import numpy as np
import pytest
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

VOCAB = 512
HIDDEN = 128
SEQ = 16
BATCH = 4


def _config(tie: bool):
    if jax.device_count() < 4:
        pytest.skip(f"needs >=4 devices for fsdp x tp2, have {jax.device_count()}")
    return ed.LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ,
        sharding_axis_dims=(1, 1, jax.device_count() // 2, 1, 2, 1),
        tie_word_embeddings=tie,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    )


def _build(config):
    with config.mesh:
        return ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )


def _kernel_seen_by_head(model, monkeypatch, run):
    """Run ``run()`` and return the kernel override the LM head's forward received."""
    head = model.get_lm_head()
    cls = type(head)
    original = cls.forward
    seen = {}

    def spy(self, inputs, w=None):
        if self is head:
            seen["w"] = w
        return original(self, inputs, w=w)

    monkeypatch.setattr(cls, "forward", spy)
    out = run()
    monkeypatch.setattr(cls, "forward", original)
    return seen.get("w"), out


def _vocab_axes(spec: PartitionSpec) -> tuple[str, ...]:
    trailing = tuple(spec)[-1] if len(tuple(spec)) == 2 else None
    return (trailing,) if isinstance(trailing, str) else tuple(trailing or ())


def test_untied_head_kernel_is_vocab_sharded_reference():
    """Reference layout the tied path must reproduce: vocab (trailing) axis on tp."""
    model = _build(_config(tie=False))
    spec = model.get_lm_head().weight.value.sharding.spec
    assert "tp" in _vocab_axes(spec), spec


@pytest.mark.parametrize("path", ["apply_lm_head", "make_lm_head_fn"])
def test_tied_kernel_matches_head_layout_and_untied_numerics(path, monkeypatch):
    config = _config(tie=True)
    model = _build(config)
    head = model.get_lm_head()
    head_spec = head.weight.value.sharding.spec
    hidden = jax.random.normal(jax.random.PRNGKey(1), (BATCH, SEQ, HIDDEN), dtype=jnp.float32)

    with config.mesh:
        if path == "apply_lm_head":
            w, logits = _kernel_seen_by_head(model, monkeypatch, lambda: model.apply_lm_head(hidden))
        else:
            w, logits = _kernel_seen_by_head(model, monkeypatch, lambda: model.make_lm_head_fn()(hidden))

    assert w is not None, "tie_word_embeddings=True but the head received no tied kernel"
    assert isinstance(w.sharding, NamedSharding), type(w.sharding)
    assert w.sharding.spec == head_spec, (
        f"[{path}] tied kernel layout {w.sharding.spec} != head kernel layout {head_spec}: the contracting "
        "dim stays tensor-sharded instead of vocab"
    )
    assert "tp" in _vocab_axes(w.sharding.spec)

    # Parity with the untied-equivalent projection: a head whose own kernel is embedding.T.
    embedding = np.asarray(model.get_embedding().weight.value)
    expected = np.einsum("bsh,vh->bsv", np.asarray(hidden), embedding)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    np.testing.assert_allclose(np.asarray(logits), expected, rtol=2e-5, atol=2e-5)
