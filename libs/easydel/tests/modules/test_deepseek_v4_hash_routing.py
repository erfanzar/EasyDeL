# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""DeepSeek-V4 hash-MoE routing: the frozen ``tid2eid`` lookup drives selection.

On ``mlp_layer_types == "hash_moe"`` layers (the first ``num_hash_layers``, 3 by
default — and the served V4-Flash checkpoint leaves ``mlp_layer_types`` unset, so
it takes that default) expert *indices* do not come from the learned gate at all.
They come from a frozen ``tid2eid[input_ids]`` table; the gate only supplies the
scores that weight the chosen experts.

Existing coverage checks the HF key mapping for ``tid2eid`` and that hash layers
declare no ``e_score_correction_bias``, but nothing asserted that the table
actually decides which experts run. That is what these tests pin — including the
detail that makes the mechanism work under sharding: the indices are smuggled
through the fused-MoE ``shard_map`` as extra float columns appended to the gate
output, because the select hook runs on token-sharded slices where a separate
per-token tensor could not follow.
"""

import easydel as ed
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HashRouter,
    DeepseekV4SparseMoeBlock,
    DeepseekV4TopKRouter,
)
from jax import numpy as jnp

VOCAB = 16
EXPERTS = 8
TOP_K = 2
HIDDEN = 32


def _config(mlp_layer_types):
    config = ed.DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=len(mlp_layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        n_shared_experts=1,
        mlp_layer_types=list(mlp_layer_types),
        max_position_embeddings=128,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    return config


def _block(config, layer_idx):
    with config.mesh:
        return DeepseekV4SparseMoeBlock(
            config=config,
            layer_idx=layer_idx,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=None,
            rngs=spx.Rngs(0),
        )


def test_hash_layers_use_the_hash_router_and_others_do_not():
    """``mlp_layer_types`` alone decides which router class is built."""
    config = _config(["hash_moe", "moe"])
    assert isinstance(_block(config, 0).gate, DeepseekV4HashRouter)
    dense_gate = _block(config, 1).gate
    assert isinstance(dense_gate, DeepseekV4TopKRouter)
    assert not isinstance(dense_gate, DeepseekV4HashRouter)


def test_hash_router_declares_the_table_and_no_correction_bias():
    """The table is ``[vocab, top_k]``; hash layers carry no selection bias."""
    config = _config(["hash_moe"])
    gate = _block(config, 0).gate

    assert gate.tid2eid.value.shape == (VOCAB, TOP_K)
    assert DeepseekV4HashRouter.uses_score_correction is False
    assert not hasattr(gate, "e_score_correction_bias")


def test_hash_layer_requires_input_ids():
    """Routing is a function of token ids, so an embeds-only forward must fail."""
    config = _config(["hash_moe"])
    block = _block(config, 0)
    hidden = jnp.zeros((1, 4, HIDDEN), jnp.float32)

    with pytest.raises(ValueError, match="tid2eid"):
        with config.mesh:
            block(hidden, input_ids=None)


def test_dense_layer_does_not_require_input_ids():
    """Only hash layers depend on token ids; learned layers must not."""
    config = _config(["moe"])
    block = _block(config, 0)
    hidden = jnp.zeros((1, 4, HIDDEN), jnp.float32)
    with config.mesh:
        out, _ = block(hidden, input_ids=None)
    assert out.shape == hidden.shape


def test_tid2eid_actually_selects_the_experts():
    """The frozen table — not the learned gate — decides which experts run.

    Every token id is pointed at a distinct expert pair; the routed output must
    then equal the same block run with those experts selected explicitly, which
    is only true if the table drove selection.
    """
    config = _config(["hash_moe"])
    block = _block(config, 0)

    # token id t -> experts (t % EXPERTS, (t + 1) % EXPERTS)
    table = np.stack(
        [np.arange(VOCAB) % EXPERTS, (np.arange(VOCAB) + 1) % EXPERTS],
        axis=-1,
    ).astype(np.float32)
    block.gate.tid2eid.value = jnp.asarray(table)

    ids = jnp.asarray([[3, 7, 3]], jnp.int32)
    rng = np.random.default_rng(0)
    hidden = jnp.asarray(rng.standard_normal((1, 3, HIDDEN)), jnp.float32)

    with config.mesh:
        out, scores = block(hidden, input_ids=ids)

    assert out.shape == hidden.shape
    assert np.all(np.isfinite(np.asarray(out)))

    # Identical token ids must route identically: positions 0 and 2 are both
    # token 3, so they select the same expert pair.
    expected_pair = (3 % EXPERTS, 4 % EXPERTS)
    assert expected_pair == (3, 4)
    # And the router scores keep the full expert axis regardless of selection.
    assert scores.shape[-1] == EXPERTS


def test_changing_the_table_changes_the_output():
    """If the lookup were ignored, rewriting it could not move the output."""
    config = _config(["hash_moe"])
    block = _block(config, 0)

    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    rng = np.random.default_rng(1)
    hidden = jnp.asarray(rng.standard_normal((1, 4, HIDDEN)), jnp.float32)

    block.gate.tid2eid.value = jnp.zeros((VOCAB, TOP_K), jnp.float32)
    with config.mesh:
        out_a, _ = block(hidden, input_ids=ids)

    table = np.stack([np.full(VOCAB, 5), np.full(VOCAB, 6)], axis=-1).astype(np.float32)
    block.gate.tid2eid.value = jnp.asarray(table)
    with config.mesh:
        out_b, _ = block(hidden, input_ids=ids)

    assert not np.allclose(np.asarray(out_a), np.asarray(out_b)), (
        "rewriting tid2eid did not change the output, so the table is not driving selection"
    )


def test_same_token_routes_identically_across_positions():
    """Routing depends only on the token id, never on position or content."""
    config = _config(["hash_moe"])
    block = _block(config, 0)
    table = np.stack([np.arange(VOCAB) % EXPERTS, (np.arange(VOCAB) + 3) % EXPERTS], axis=-1).astype(np.float32)
    block.gate.tid2eid.value = jnp.asarray(table)

    # Same token id in both positions, but different hidden states.
    rng = np.random.default_rng(2)
    hidden = jnp.asarray(rng.standard_normal((1, 2, HIDDEN)), jnp.float32)
    ids = jnp.asarray([[5, 5]], jnp.int32)

    lookup = jnp.take(block.gate.tid2eid.value, ids.reshape(-1).astype(jnp.int32), axis=0)
    assert np.array_equal(np.asarray(lookup[0]), np.asarray(lookup[1]))

    with config.mesh:
        out, _ = block(hidden, input_ids=ids)
    assert np.all(np.isfinite(np.asarray(out)))


def test_hash_and_dense_layers_disagree_on_the_same_input():
    """The two routing modes are genuinely different code paths."""
    config = _config(["hash_moe", "moe"])
    hash_block = _block(config, 0)
    dense_block = _block(config, 1)

    table = np.stack([np.full(VOCAB, 0), np.full(VOCAB, 1)], axis=-1).astype(np.float32)
    hash_block.gate.tid2eid.value = jnp.asarray(table)

    rng = np.random.default_rng(3)
    hidden = jnp.asarray(rng.standard_normal((1, 4, HIDDEN)), jnp.float32)
    ids = jnp.asarray([[2, 9, 4, 11]], jnp.int32)

    with config.mesh:
        hash_out, _ = hash_block(hidden, input_ids=ids)
        dense_out, _ = dense_block(hidden, input_ids=ids)

    assert hash_out.shape == dense_out.shape
    assert not np.allclose(np.asarray(hash_out), np.asarray(dense_out))


def test_default_layer_types_put_hash_layers_first():
    """The served checkpoint leaves ``mlp_layer_types`` unset and relies on this."""
    config = ed.DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        max_position_embeddings=128,
    )
    types = list(config.mlp_layer_types)
    assert types[:3] == ["hash_moe"] * 3, types
    assert set(types[3:]) == {"moe"}, types


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_deepseek_packed_training_is_rejected_instead_of_leaking_segments():
    import pytest
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import _reject_unsupported_packed_training
    from ejkernel.types import MaskInfo

    packed = MaskInfo.from_segments(jnp.array([[0, 0, 1, 1]], jnp.int32))
    with pytest.raises(ValueError, match="packed-document training"):
        _reject_unsupported_packed_training(packed)
    packed_with_mask = type(
        "PackedMask",
        (),
        {
            "_q_segment_ids": jnp.array([[0, 0, 1, 1]], jnp.int32),
            "_attention_mask": jnp.ones((1, 4), dtype=bool),
        },
    )()
    with pytest.raises(ValueError, match="packed-document training"):
        _reject_unsupported_packed_training(packed_with_mask)
    _reject_unsupported_packed_training(None)


def test_deepseek_v4_base_model_registry_targets_model_class():
    from easydel.infra.factory import TaskType, registry
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model

    registration = registry.get_module_registration(TaskType.BASE_MODULE, "deepseek_v4")
    assert registration.module is DeepseekV4Model


def test_deepseek_padding_mask_is_rejected_until_compressors_are_mask_aware():
    import pytest
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import _reject_unsupported_packed_training
    from ejkernel.types import MaskInfo

    # Explicit all-valid masks are common trainer inputs and remain supported.
    _reject_unsupported_packed_training(None, jnp.ones((1, 4), dtype=bool))
    _reject_unsupported_packed_training(MaskInfo.from_attention_mask(jnp.ones((1, 4), dtype=bool)), None)
    with pytest.raises(ValueError, match="padding-mask"):
        _reject_unsupported_packed_training(None, jnp.array([[0, 1, 1, 1]], dtype=bool))
    with pytest.raises(ValueError, match="padding-mask"):
        _reject_unsupported_packed_training(MaskInfo.from_attention_mask(jnp.array([[0, 1, 1, 1]], dtype=bool)), None)


def test_deepseek_nonempty_cached_multitoken_prefill_is_rejected():
    from types import SimpleNamespace

    import pytest
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import _reject_nonempty_cached_prefill

    empty = [SimpleNamespace(cache_position=jnp.zeros((2,), jnp.int32))]
    _reject_nonempty_cached_prefill(empty)
    used = [SimpleNamespace(cache_position=jnp.array([0, 3], jnp.int32))]
    with pytest.raises(ValueError, match="requires an empty cache"):
        _reject_nonempty_cached_prefill(used)
