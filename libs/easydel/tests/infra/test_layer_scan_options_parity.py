"""Models driven by ``layer_scan_options`` match the default scan lowering in loss and gradients (TPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import spectrax as spx

VOCAB, SEQ, BATCH = 256, 32, 4


def _build(family: str):
    common = dict(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        scan_layers=True,
    )
    if family == "llama":
        from easydel.modules.llama import LlamaConfig as Config
        from easydel.modules.llama import LlamaForCausalLM as Model
    else:
        from easydel.modules.gemma3 import Gemma3ForCausalLM as Model
        from easydel.modules.gemma3 import Gemma3TextConfig as Config

        common.update(sliding_window=8, sliding_window_pattern=2)
    config = Config(**common)
    config.attn_dtype = jnp.float32
    model = Model(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    return model.shard_model()


def _loss_and_grads(model, options, ids):
    model.config.layer_scan_options = options
    gdef, state = spx.export(model)

    def loss(st):
        logits = spx.bind(gdef, st)(input_ids=ids).logits.astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], ids[:, 1:]).mean()

    with model.mesh:
        value, grads = jax.jit(jax.value_and_grad(loss, allow_int=True))(state)
    leaves = [g for g in jax.tree.leaves(grads) if jnp.issubdtype(g.dtype, jnp.inexact)]
    return float(value), leaves


@pytest.mark.parametrize("family", ["llama", "gemma3"])
@pytest.mark.parametrize(
    "options",
    [
        None,
        {"barrier": True},
        {"block_scan": True},
        {"prefetch_axes": "fsdp"},
        {"text": {"prefetch_axes": "fsdp", "barrier": True, "prefetch_defer_dw": True}},
        {"vision": {"block_scan": True}},
    ],
)
def test_scan_options_match_default_lowering(family, options):
    model = _build(family)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (BATCH, SEQ)), jnp.int32)
    ref_loss, ref_grads = _loss_and_grads(model, {}, ids)
    loss, grads = _loss_and_grads(model, options, ids)
    np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
    assert len(grads) == len(ref_grads)
    for g, r in zip(grads, ref_grads, strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=2e-4, atol=1e-6)
