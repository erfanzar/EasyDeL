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

"""DSpark vocab-IO init from tied targets, packed-document block masks, and anchor sampling."""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sequence_packing import fold_sequence_packing_segments
from easydel.modules._drafting import create_dspark_attention_mask, sample_anchor_positions


def _llama(tie: bool):
    cfg = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=tie,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
    )
    return cfg, ed.LlamaForCausalLM(config=cfg, rngs=spx.Rngs(1), dtype=jnp.float32, param_dtype=jnp.float32)


def _dspark(**overrides):
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        target_hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        target_layer_ids=(-1, 0),
        block_size=3,
        num_anchors=2,
        mask_token_id=63,
        markov_rank=4,
    )
    kwargs.update(overrides)
    return ed.DSparkModel(config=ed.DSparkConfig(**kwargs), rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def test_untied_drafter_head_is_seeded_from_tied_target_embedding():
    cfg, target = _llama(tie=True)
    with cfg.mesh:
        drafter = _dspark()
        drafter.initialize_embeddings_and_head_from_target(target)
        embed = np.asarray(target.get_embedding().weight.value)
        stale_head = np.asarray(target.get_lm_head().weight.value)
        head = np.asarray(drafter.lm_head.weight.value)
    np.testing.assert_array_equal(np.asarray(drafter.embed_tokens.weight.value), embed)
    np.testing.assert_array_equal(head, embed.T)
    assert not np.array_equal(head, stale_head)


def test_untied_target_head_array_is_copied():
    cfg, target = _llama(tie=False)
    with cfg.mesh:
        drafter = _dspark()
        drafter.initialize_embeddings_and_head_from_target(target)
        target_head = np.asarray(target.get_lm_head().weight.value)
        head = np.asarray(drafter.lm_head.weight.value)
    np.testing.assert_array_equal(head, target_head)


def test_vocab_gate_rejects_drifted_copies():
    cfg, target = _llama(tie=True)
    with cfg.mesh:
        drafter = _dspark()
        drafter.initialize_embeddings_and_head_from_target(target)
        drafter.check_embeddings_and_head_against_target(target)
        drafter.lm_head.weight.value = drafter.lm_head.weight.value.at[0, 0].add(1.0)
        with pytest.raises(RuntimeError, match="lm_head"):
            drafter.check_embeddings_and_head_against_target(target)
        drafter.lm_head.weight.value = drafter.lm_head.weight.value.at[0, 0].add(-1.0)
        drafter.embed_tokens.weight.value = drafter.embed_tokens.weight.value.at[3, 1].add(0.5)
        with pytest.raises(RuntimeError, match="embed_tokens"):
            drafter.check_embeddings_and_head_against_target(target)


def test_block_mask_restricts_context_to_anchor_document():
    anchors = jnp.array([[2, 5]], dtype=jnp.int32)
    keep = jnp.ones((1, 2), dtype=jnp.bool_)
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    packed = np.asarray(
        create_dspark_attention_mask(
            anchor_positions=anchors, block_keep_mask=keep, seq_len=8, block_size=2, segment_ids=segment_ids
        )
    )
    unpacked = np.asarray(
        create_dspark_attention_mask(anchor_positions=anchors, block_keep_mask=keep, seq_len=8, block_size=2)
    )
    assert packed.shape == unpacked.shape == (1, 4, 12)
    expected_packed = np.zeros((4, 8), dtype=bool)
    expected_packed[:2, [0, 1]] = True
    expected_packed[2:, [4]] = True
    expected_unpacked = np.zeros((4, 8), dtype=bool)
    expected_unpacked[:2, :2] = True
    expected_unpacked[2:, :5] = True
    np.testing.assert_array_equal(packed[0, :, :8], expected_packed)
    np.testing.assert_array_equal(unpacked[0, :, :8], expected_unpacked)
    np.testing.assert_array_equal(packed[0, :, 8:], unpacked[0, :, 8:])


def _reference_anchors(loss_mask, num_anchors):
    """The anchor rule before ``salt`` / ``fold_batch_index`` / ``label_horizon`` existed."""
    batch_size, seq_len = loss_mask.shape
    n = seq_len - 1
    valid = (loss_mask[:, :n] > 0.5) & (loss_mask[:, 1 : n + 1] > 0.5)
    cand = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32)[None, :], (batch_size, n))
    rows = jnp.arange(batch_size, dtype=jnp.uint32)[:, None]
    s = (cand.astype(jnp.uint32) ^ jnp.uint32(0x9E3779B9)) * jnp.uint32(0x85EBCA6B)
    s = (s ^ (s >> jnp.uint32(13))) * jnp.uint32(0xC2B2AE35)
    s = s ^ (s >> jnp.uint32(16)) ^ (rows * jnp.uint32(0x27D4EB2D))
    order = jnp.argsort(jnp.where(valid, s, jnp.uint32(0xFFFFFFFF)), axis=1)
    picked = jnp.take_along_axis(cand, order, axis=1)[:, :num_anchors]
    kept = jnp.take_along_axis(valid, order, axis=1)[:, :num_anchors]
    anchors = jnp.sort(jnp.where(kept, picked, seq_len + 1), axis=1)
    kept = anchors < seq_len
    return jnp.where(kept, anchors, 0), kept


def test_anchor_defaults_reproduce_previous_rule():
    loss_mask = (jax.random.uniform(jax.random.PRNGKey(3), (3, 24)) > 0.3).astype(jnp.float32)
    anchors, keep = sample_anchor_positions(loss_mask, num_anchors=5)
    ref_anchors, ref_keep = _reference_anchors(loss_mask, 5)
    np.testing.assert_array_equal(np.asarray(anchors), np.asarray(ref_anchors))
    np.testing.assert_array_equal(np.asarray(keep), np.asarray(ref_keep))


def test_anchor_label_horizon_requires_full_supervised_window():
    loss_mask = jnp.array([[1, 1, 1, 1, 0, 1, 1, 0]], dtype=jnp.float32)
    anchors, keep = sample_anchor_positions(loss_mask, num_anchors=8)
    assert sorted(np.asarray(anchors)[np.asarray(keep)].tolist()) == [0, 1, 2, 5]
    anchors, keep = sample_anchor_positions(loss_mask, num_anchors=8, label_horizon=3)
    assert np.asarray(anchors)[np.asarray(keep)].tolist() == [0]
    with pytest.raises(ValueError, match="label_horizon"):
        sample_anchor_positions(loss_mask, num_anchors=2, label_horizon=0)


def test_salted_anchors_without_row_index_do_not_depend_on_microbatch_split():
    loss_mask = jnp.ones((4, 32), dtype=jnp.float32)
    salt = jnp.array([7, 11, 7, 13], dtype=jnp.uint32)
    full, _ = sample_anchor_positions(loss_mask, num_anchors=4, salt=salt, fold_batch_index=False)
    tail, _ = sample_anchor_positions(loss_mask[2:], num_anchors=4, salt=salt[2:], fold_batch_index=False)
    np.testing.assert_array_equal(np.asarray(full[2:]), np.asarray(tail))
    np.testing.assert_array_equal(np.asarray(full[0]), np.asarray(full[2]))
    assert not np.array_equal(np.asarray(full[0]), np.asarray(full[1]))


def test_dspark_block_forward_reads_packed_segments_from_mask_info():
    model = _dspark()
    input_ids = (jnp.arange(16, dtype=jnp.int32).reshape(2, 8) * 5) % 60
    key = jax.random.PRNGKey(0)
    hidden = tuple(jax.random.normal(jax.random.fold_in(key, i), (2, 8, 32), dtype=jnp.float32) for i in range(2))
    loss_mask = jnp.ones((2, 8), dtype=jnp.float32)
    anchors = jnp.array([[1, 5], [2, 6]], dtype=jnp.int32)
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1]], dtype=jnp.int32)

    def run(**kw):
        return np.asarray(
            model(
                input_ids=input_ids, target_hidden_states=hidden, loss_mask=loss_mask, anchor_positions=anchors, **kw
            ).block_logits
        )

    unpacked = run()
    single_doc = run(segment_ids=jnp.zeros_like(segment_ids))
    direct = run(segment_ids=segment_ids)
    folded = fold_sequence_packing_segments({"input_ids": input_ids, "segment_ids": segment_ids})
    via_mask_info = run(mask_info=folded["mask_info"])
    np.testing.assert_array_equal(single_doc, unpacked)
    np.testing.assert_allclose(via_mask_info, direct, rtol=1e-6, atol=1e-6)
    # First-document blocks see the same context either way; second-document blocks lose the other document.
    np.testing.assert_allclose(direct[:, 0], unpacked[:, 0], rtol=1e-5, atol=1e-5)
    assert np.max(np.abs(direct[:, 1] - unpacked[:, 1])) > 1e-4
