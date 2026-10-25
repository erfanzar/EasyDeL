"""Loading a Hugging Face checkpoint with ``scan_layers=True`` materializes the stacked layers (TPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

VOCAB, SEQ = 256, 16


@pytest.fixture(scope="module")
def hf_checkpoint(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ,
        tie_word_embeddings=False,
    )
    path = tmp_path_factory.mktemp("tiny_llama")
    transformers.LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


def _load(path, scan):
    import easydel as ed

    return ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        auto_shard_model=True,
        config_kwargs={"scan_layers": scan, "attn_dtype": jnp.float32},
    )


def test_scanned_load_matches_unscanned_logits(hf_checkpoint):
    plain = _load(hf_checkpoint, scan=False)
    scanned = _load(hf_checkpoint, scan=True)
    assert isinstance(scanned.model.layers, spx.nn.StackedModuleList)
    assert len(scanned.model.layers) == 3
    assert not scanned.abstract_parameter_leaves()

    ids = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (2, SEQ)), jnp.int32)
    with plain.mesh:
        ref = np.asarray(plain(input_ids=ids).logits)
    with scanned.mesh:
        out = np.asarray(scanned(input_ids=ids).logits)
    assert np.abs(ref).max() > 0
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
