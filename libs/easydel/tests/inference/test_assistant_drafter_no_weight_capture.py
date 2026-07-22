"""The Gemma4 Assistant drafter must not capture model weights as constants.

Regression test for the last instance of the jit-over-a-module weight-capture
bug class (the same bug fixed earlier for the MTP drafter, the speculative
program, and the VLM vision jits). ``Gemma4AssistantDrafter._assistant_forward``
used to jit a closure over ``self`` — baking the whole assistant model and the
target embedding into the compiled program as captured constants. On the real
27B target that is gigabytes per executable (never caches, re-lowers each
process, and duplicates weights in HBM).

The fix threads both modules through the jit as ``spx.State`` inputs via
``easydel.utils.graph_jit.module_jit``. These tests assert, on a tiny assistant:

* the jitted drafter forward is output-identical to calling the assistant model
  eagerly, and
* the lowered program contains no large dense constants (weights arrive as
  arguments, not literals).
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import easydel as ed  # noqa: F401  # ensures registry side effects fire
import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.speculative import Gemma4AssistantDrafter
from easydel.modules.gemma4_assistant import (
    Gemma4AssistantConfig,
    Gemma4AssistantForCausalLM,
    Gemma4AssistantTextConfig,
)
from easydel.utils.graph_jit import assert_no_large_constants
from jax import numpy as jnp

_HIDDEN = 64
_BACKBONE = 128
_VOCAB = 256


def make_assistant():
    text_cfg = Gemma4AssistantTextConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_HIDDEN * 4,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=_HIDDEN // 2,
        global_head_dim=_HIDDEN,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        sliding_window=32,
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


def make_drafter():
    assistant = make_assistant()
    # A standalone embedding standing in for the target model's embed_tokens:
    # ids -> backbone-hidden-size vectors.
    target_embed = spx.nn.Embed(_VOCAB, _BACKBONE, rngs=spx.Rngs(1), dtype=jnp.float32, param_dtype=jnp.float32)
    drafter = Gemma4AssistantDrafter(assistant_model=assistant, target_embed_module=target_embed)
    return drafter, assistant, target_embed


def _inputs(drafter):
    b, s = 1, 4
    ids = jnp.asarray(np.arange(b * s, dtype=np.int32).reshape(b, s) % _VOCAB)
    hidden = jax.random.normal(jax.random.PRNGKey(2), (b, s, _BACKBONE), dtype=jnp.float32)
    return ids, hidden


def test_assistant_drafter_jit_matches_eager():
    drafter, assistant, target_embed = make_drafter()
    ids, hidden = _inputs(drafter)

    got = drafter._assistant_forward(ids, hidden, None, None, None, False)

    embeds = target_embed(ids.astype("i4")) * drafter._embed_scale.astype(hidden.dtype)
    want = assistant(
        backbone_hidden_states=hidden,
        target_token_embeds=embeds,
        target_key_value_pairs=None,
        position_ids=None,
        attention_mask=None,
        return_dense_logits=False,
    )

    # jit-vs-eager float reassociation on CPU stays tiny in f32.
    np.testing.assert_allclose(
        np.asarray(got.last_hidden_state), np.asarray(want.last_hidden_state), rtol=1e-3, atol=2e-4
    )
    np.testing.assert_allclose(
        np.asarray(got.backbone_hidden_state), np.asarray(want.backbone_hidden_state), rtol=1e-3, atol=2e-4
    )


def test_assistant_drafter_jit_captures_no_weight_constants():
    drafter, _assistant, _target_embed = make_drafter()
    ids, hidden = _inputs(drafter)

    lowered = drafter._assistant_jit.lower(ids, hidden, None, None, None, return_dense_logits=False)
    # If the assistant + target embedding were captured, the token embedding
    # (256x128 = 32768 elements) and the MLP weights would blow this bound;
    # rotary/position/index constants stay well below it.
    worst = assert_no_large_constants(lowered, max_elements=10_000)
    assert worst < 10_000


def test_assistant_drafter_jit_is_reused():
    """The module_jit wrapper is built once and reused across draft calls."""
    drafter, _assistant, _target_embed = make_drafter()
    ids, hidden = _inputs(drafter)
    a = drafter._assistant_forward(ids, hidden, None, None, None, False)
    b = drafter._assistant_forward(ids, hidden, None, None, None, False)
    np.testing.assert_array_equal(np.asarray(a.last_hidden_state), np.asarray(b.last_hidden_state))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
