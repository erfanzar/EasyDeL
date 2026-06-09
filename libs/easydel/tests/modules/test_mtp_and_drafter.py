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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness tests for Qwen3.5 MTP + Gemma4 Assistant + speculative primitives.

Standalone runner — not wired into pytest harness because the existing
EasyDeL tester compares against torch HF impls that don't yet ship
Gemma4Assistant. These tests focus on:

- Shape/dtype correctness of forward passes on tiny configs
- Compute_mtp_loss math sanity
- Centroid head selection invariants (top-k centroids, valid token IDs)
- Speculative-decoding primitives (accept/reject + resample) numerically
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx

import easydel as ed  # ensures registry side effects fire

__test__ = False

from easydel.modules.gemma4_assistant import (
    Gemma4AssistantCentroidHead,
    Gemma4AssistantConfig,
    Gemma4AssistantForCausalLM,
    Gemma4AssistantTextConfig,
)
from easydel.modules.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MTPHead,
    Qwen3_5TextConfig,
)

_RESULTS: list[tuple[str, str, str]] = []


def test(name: str):
    def deco(fn):
        def wrapper():
            t0 = time.time()
            try:
                fn()
                ms = (time.time() - t0) * 1000
                _RESULTS.append(("PASS", name, f"{ms:.1f} ms"))
                print(f"  PASS [{ms:7.1f} ms] {name}")
                return True
            except Exception:
                tb = traceback.format_exc()
                first = tb.strip().splitlines()[-1]
                _RESULTS.append(("FAIL", name, first))
                print(f"  FAIL              {name}")
                print(f"    -> {first}")
                for line in tb.strip().splitlines()[-12:]:
                    print(f"       {line}")
                return False

        wrapper.__name__ = fn.__name__
        return wrapper

    return deco


def assert_close(a, b, atol=1e-5, rtol=1e-5, msg=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg}: max abs diff {max_abs} > atol={atol}")


def assert_shape(arr, expected: tuple[int, ...], name: str = "tensor"):
    got = tuple(arr.shape)
    if got != expected:
        raise AssertionError(f"{name} shape mismatch: got {got}, expected {expected}")


def make_qwen35_text_config(*, mtp_layers: int = 1, hidden: int = 64, vocab: int = 128):
    return Qwen3_5TextConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden // 2,
        max_position_embeddings=64,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        mtp_num_hidden_layers=mtp_layers,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
    )


def make_gemma4_assistant_config(
    *, hidden: int = 64, backbone_hidden: int = 128, vocab: int = 256, num_centroids: int = 16
):
    text_cfg = Gemma4AssistantTextConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 4,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=hidden // 2,
        global_head_dim=hidden,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        sliding_window=32,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    return Gemma4AssistantConfig(
        text_config=text_cfg,
        backbone_hidden_size=backbone_hidden,
        num_centroids=num_centroids,
        centroid_intermediate_top_k=4,
        use_ordered_embeddings=True,
        tie_word_embeddings=True,
    )


@test("Qwen3_5MTPHead.forward shape + dtype")
def test_qwen35_mtp_forward_shape():
    cfg = make_qwen35_text_config()
    rngs = spx.Rngs(0)
    head = Qwen3_5MTPHead(config=cfg, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    B, S, H = 2, 8, cfg.hidden_size
    key = jax.random.PRNGKey(0)
    prev = jax.random.normal(key, (B, S, H)).astype(jnp.bfloat16)
    nxt = jax.random.normal(jax.random.split(key)[1], (B, S, H)).astype(jnp.bfloat16)
    out = head(prev_hidden_states=prev, next_token_embeds=nxt)
    assert_shape(out.last_hidden_state, (B, S, H), "mtp.last_hidden_state")
    assert out.last_hidden_state.dtype == jnp.bfloat16, f"got dtype {out.last_hidden_state.dtype}"
    assert not jnp.any(jnp.isnan(out.last_hidden_state)), "MTP output contains NaN"


@test("Qwen3_5MTPHead handles batch=1 + single position (decode mode)")
def test_qwen35_mtp_single_token():
    cfg = make_qwen35_text_config()
    head = Qwen3_5MTPHead(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    B, S, H = 1, 1, cfg.hidden_size
    prev = jnp.zeros((B, S, H), jnp.float32)
    nxt = jnp.zeros((B, S, H), jnp.float32)
    out = head(prev_hidden_states=prev, next_token_embeds=nxt)
    assert_shape(out.last_hidden_state, (B, S, H), "mtp.last_hidden_state (decode)")


@test("Qwen3_5ForCausalLM.compute_mtp_outputs end-to-end (tiny)")
def test_qwen35_for_causal_lm_mtp_e2e():
    cfg = make_qwen35_text_config()
    rngs = spx.Rngs(0)
    model = Qwen3_5ForCausalLM(config=cfg, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    assert model.has_mtp(), "Qwen3.5 model should have MTP head when mtp_num_hidden_layers > 0"
    B, S, H = 1, 4, cfg.hidden_size
    last_h = jax.random.normal(jax.random.PRNGKey(1), (B, S, H)).astype(jnp.bfloat16)
    input_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)

    class FakeOutputs:
        def __init__(self, lh):
            self.last_hidden_state = lh

    mtp_out = model.compute_mtp_outputs(FakeOutputs(last_h), input_ids=input_ids)
    assert mtp_out is not None, "compute_mtp_outputs returned None despite has_mtp"
    assert_shape(mtp_out.last_hidden_state, (B, S, H), "mtp_output.last_hidden_state")
    logits = model.compute_mtp_logits(mtp_out)
    assert_shape(logits, (B, S, cfg.vocab_size), "mtp_logits")


@test("Qwen3_5ForCausalLM.compute_mtp_outputs returns None when MTP disabled")
def test_qwen35_no_mtp_when_disabled():
    cfg = make_qwen35_text_config(mtp_layers=0)
    model = Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0))
    assert not model.has_mtp(), "has_mtp() should be False when mtp_num_hidden_layers == 0"
    assert model.mtp is None, "model.mtp should be None"


@test("TRAIN: forward folds MTP loss into outputs.aux_loss (the trainer channel)")
def test_mtp_forward_populates_aux_loss():
    """The trainer's compute_loss adds outputs.aux_loss to the loss.
    With mtp_loss_coef>0 the forward must populate aux_loss with the
    MTP CE loss; with coef=0 it must NOT (clean opt-out)."""
    ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)

    cfg_on = make_qwen35_text_config(mtp_layers=1)
    cfg_on.mtp_loss_coef = 0.3
    model_on = Qwen3_5ForCausalLM(config=cfg_on, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    out_on = model_on(input_ids=ids)
    assert getattr(out_on, "aux_loss", None) is not None, (
        "aux_loss is None with mtp_loss_coef>0 — MTP loss not folded into the trainer channel"
    )
    assert float(out_on.aux_loss) > 0.0, f"MTP aux_loss should be positive, got {float(out_on.aux_loss)}"
    print(f"       mtp_loss_coef=0.3 -> outputs.aux_loss = {float(out_on.aux_loss):.4f}")

    cfg_off = make_qwen35_text_config(mtp_layers=1)
    cfg_off.mtp_loss_coef = 0.0
    model_off = Qwen3_5ForCausalLM(config=cfg_off, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    out_off = model_off(input_ids=ids)
    off_aux = getattr(out_off, "aux_loss", None)
    assert off_aux is None or float(off_aux) == 0.0, f"aux_loss should be None/0 with mtp_loss_coef=0, got {off_aux}"
    print(f"       mtp_loss_coef=0.0 -> outputs.aux_loss = {off_aux} (MTP frozen, as designed)")


@test("TRAIN: MTP head receives non-zero gradients via the real trainer loss path")
def test_mtp_head_receives_gradients():
    """The decisive trainability proof. Replicates the trainer's exact
    gradient computation: split_module -> merge_module -> compute_loss
    -> outputs.loss -> jax.value_and_grad w.r.t. parameters. Asserts
    the mtp.* parameters get non-zero gradient with mtp_loss_coef>0,
    and EXACTLY-zero gradient with mtp_loss_coef=0 (negative control:
    MTP params are not in the loss graph at all)."""
    ids = jnp.array([[3, 1, 4, 1, 5, 9, 2, 6]], dtype=jnp.int32)

    def _path_str(path):
        parts = []
        for p in path:
            if hasattr(p, "key"):
                parts.append(str(p.key))
            elif hasattr(p, "idx"):
                parts.append(str(p.idx))
            else:
                parts.append(str(p))
        return parts

    def measure(coef):
        cfg = make_qwen35_text_config(mtp_layers=1)
        cfg.mtp_loss_coef = coef
        model = Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
        gdef, gstate, gother = model.split_module()

        def loss_fn(gs):
            m = model.merge_module(gdef, gs, gother)
            outputs, _metrics = m.compute_loss(input_ids=ids, labels=ids)
            return outputs.loss

        loss, grads = jax.value_and_grad(loss_fn)(gstate)
        mtp_sq, main_sq = 0.0, 0.0
        for path, g in jax.tree_util.tree_flatten_with_path(grads)[0]:
            parts = _path_str(path)
            gsq = float(jnp.sum(jnp.asarray(g, jnp.float32) ** 2))
            if "mtp" in parts:
                mtp_sq += gsq
            else:
                main_sq += gsq
        return float(loss), mtp_sq**0.5, main_sq**0.5

    loss_on, mtp_norm_on, main_norm_on = measure(0.3)
    print(f"       coef=0.3:  loss={loss_on:.4f}  |grad(mtp.*)|={mtp_norm_on:.5f}  |grad(main)|={main_norm_on:.5f}")
    assert mtp_norm_on > 1e-6, (
        f"MTP params got ZERO gradient with mtp_loss_coef=0.3 — the head would NOT train. |grad(mtp)|={mtp_norm_on}"
    )
    assert main_norm_on > 1e-6, "main model got zero gradient — something is wrong with the loss"

    loss_off, mtp_norm_off, main_norm_off = measure(0.0)
    print(f"       coef=0.0:  loss={loss_off:.4f}  |grad(mtp.*)|={mtp_norm_off:.5f}  |grad(main)|={main_norm_off:.5f}")
    assert mtp_norm_off < 1e-9, (
        f"MTP params got NON-zero gradient with mtp_loss_coef=0 — opt-out is broken. |grad(mtp)|={mtp_norm_off}"
    )
    print("       => MTP head trains when coef>0, is frozen when coef=0 (real value_and_grad proof)")


@test("compute_mtp_loss: shift-by-2 alignment + ignore_index correctness")
def test_compute_mtp_loss_math():
    """Verify MTP loss = CE(logits[t], labels[t+2]) with trailing -100."""
    cfg = make_qwen35_text_config()
    model = Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    B, S, V = 1, 5, cfg.vocab_size
    labels = jnp.array([[10, 20, 30, 40, 50]], dtype=jnp.int32)
    logits = jnp.full((B, S, V), -100.0, dtype=jnp.float32)
    logits = logits.at[0, 0, 30].set(0.0)
    logits = logits.at[0, 1, 40].set(0.0)
    logits = logits.at[0, 2, 50].set(0.0)
    logits = logits.at[0, 3, 50].set(0.0)  # ignored
    logits = logits.at[0, 4, 50].set(0.0)  # ignored
    loss = model.compute_mtp_loss(logits, labels)
    assert float(loss) < 1e-3, f"Loss should be ~0 for perfect predictions, got {float(loss)}"

    bad_logits = jnp.full((B, S, V), -100.0, dtype=jnp.float32)
    bad_logits = bad_logits.at[0, 0, 99].set(0.0)  # predicts 99, target is 30
    bad_logits = bad_logits.at[0, 1, 99].set(0.0)  # predicts 99, target is 40
    bad_logits = bad_logits.at[0, 2, 99].set(0.0)  # predicts 99, target is 50
    bad_loss = model.compute_mtp_loss(bad_logits, labels)
    assert float(bad_loss) > 50.0, f"Loss should be large for wrong predictions, got {float(bad_loss)}"


@test("compute_mtp_loss: attention_mask zeroes out target positions")
def test_compute_mtp_loss_attention_mask():
    cfg = make_qwen35_text_config()
    model = Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    B, S, V = 1, 5, cfg.vocab_size
    labels = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    logits = jax.random.normal(jax.random.PRNGKey(0), (B, S, V))
    am = jnp.zeros((B, S), dtype=jnp.int32)
    loss = model.compute_mtp_loss(logits, labels, attention_mask=am)
    assert float(loss) == 0.0, f"All-masked loss should be 0, got {float(loss)}"


@test("Gemma4AssistantCentroidHead: clusters partition + top-k valid IDs")
def test_gemma4_centroid_head():
    cfg = make_gemma4_assistant_config(hidden=32, vocab=64, num_centroids=8)
    text_cfg = cfg.text_config
    rngs = spx.Rngs(0)
    head = Gemma4AssistantCentroidHead(config=cfg, rngs=rngs, dtype=jnp.float32, param_dtype=jnp.float32)
    expected_ordering = np.arange(text_cfg.vocab_size)
    np.testing.assert_array_equal(np.asarray(head.token_ordering.value), expected_ordering)
    assert head.tokens_per_centroid == 64 // 8, f"tokens_per_centroid={head.tokens_per_centroid}"

    B, S, H = 1, 3, text_cfg.hidden_size
    hidden = jax.random.normal(jax.random.PRNGKey(7), (B, S, H))
    embed_w = jax.random.normal(jax.random.PRNGKey(8), (text_cfg.vocab_size, H))
    top_logits, top_ids, _dense = head(hidden, embed_w, return_dense_logits=False)
    K = head.top_k * head.tokens_per_centroid
    assert_shape(top_logits, (B, S, K), "top_logits")
    assert_shape(top_ids, (B, S, K), "top_token_ids")
    ids = np.asarray(top_ids)
    assert ids.min() >= 0 and ids.max() < text_cfg.vocab_size, (
        f"token IDs out of range: min={ids.min()} max={ids.max()} vocab={text_cfg.vocab_size}"
    )


@test("Gemma4AssistantCentroidHead: dense logits have -inf outside selected candidates")
def test_gemma4_centroid_dense_scatter():
    cfg = make_gemma4_assistant_config(hidden=32, vocab=64, num_centroids=8)
    head = Gemma4AssistantCentroidHead(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    B, S, H = 1, 2, cfg.text_config.hidden_size
    hidden = jax.random.normal(jax.random.PRNGKey(0), (B, S, H))
    embed_w = jax.random.normal(jax.random.PRNGKey(1), (cfg.text_config.vocab_size, H))
    top_logits, top_ids, dense = head(hidden, embed_w, return_dense_logits=True)
    assert dense is not None, "dense_logits should be non-None when requested"
    K = head.top_k * head.tokens_per_centroid
    for b in range(B):
        for s in range(S):
            finite = np.isfinite(np.asarray(dense[b, s])).sum()
            assert finite == K, f"position ({b},{s}): {finite} finite entries, expected {K}"
            sel_ids = np.asarray(top_ids[b, s])
            sel_vals = np.asarray(dense[b, s])[sel_ids]
            assert_close(sel_vals, np.asarray(top_logits[b, s]), atol=1e-5, msg="dense scatter mismatch")


@test("Gemma4AssistantForCausalLM.forward shape (self-K/V fallback)")
def test_gemma4_assistant_forward_shape():
    cfg = make_gemma4_assistant_config()
    model = Gemma4AssistantForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    B, S = 1, 4
    BH = cfg.backbone_hidden_size
    H = cfg.text_config.hidden_size
    K = model.masked_embedding.top_k * model.masked_embedding.tokens_per_centroid

    backbone_hidden = jax.random.normal(jax.random.PRNGKey(0), (B, S, BH)).astype(jnp.float32)
    target_embeds = jax.random.normal(jax.random.PRNGKey(1), (B, S, BH)).astype(jnp.float32)
    out = model(
        backbone_hidden_states=backbone_hidden,
        target_token_embeds=target_embeds,
        target_key_value_pairs=None,  # fall back to self-K/V
        attention_mask=None,
        return_dense_logits=False,
    )
    assert_shape(out.last_hidden_state, (B, S, H), "drafter.last_hidden_state")
    assert_shape(out.backbone_hidden_state, (B, S, BH), "drafter.backbone_hidden_state")
    assert out.top_logits is not None, "centroid head should produce top_logits"
    assert_shape(out.top_logits, (B, S, K), "drafter.top_logits")
    assert_shape(out.top_token_ids, (B, S, K), "drafter.top_token_ids")


@test("speculative.accept_or_reject: target_p >> draft_p always accepts")
def test_accept_or_reject_dominant_target():
    from easydel.inference.speculative import accept_or_reject

    dlp = jnp.log(jnp.full((128,), 0.1))
    tlp = jnp.log(jnp.full((128,), 0.9))
    mask = accept_or_reject(dlp, tlp, jax.random.PRNGKey(0))
    assert int(jnp.sum(mask)) == 128, f"All 128 should accept when target dominates, got {int(jnp.sum(mask))}"


@test("speculative.accept_or_reject: target_p << draft_p mostly rejects")
def test_accept_or_reject_dominated_target():
    from easydel.inference.speculative import accept_or_reject

    dlp = jnp.log(jnp.full((4096,), 0.9))
    tlp = jnp.log(jnp.full((4096,), 0.1))
    mask = accept_or_reject(dlp, tlp, jax.random.PRNGKey(0))
    accept_rate = float(jnp.mean(mask))
    assert 0.08 < accept_rate < 0.15, f"Expected ~0.11 acceptance, got {accept_rate}"


@test("speculative.resample_rejected: produces valid token IDs from residual distribution")
def test_resample_rejected():
    from easydel.inference.speculative import resample_rejected

    V = 32
    tgt = jnp.full((1, V), -10.0).at[:, 5].set(0.0)
    drf = jnp.full((1, V), -10.0).at[:, 0].set(0.0)
    tlp = jax.nn.log_softmax(tgt, axis=-1)
    dlp = jax.nn.log_softmax(drf, axis=-1)
    samples = np.zeros((256,), dtype=np.int32)
    for i in range(256):
        s = resample_rejected(tlp, dlp, jax.random.PRNGKey(i))
        samples[i] = int(s[0])
    most_common = int(np.bincount(samples).argmax())
    assert most_common == 5, f"Expected most-sampled token = 5, got {most_common} (dist: {np.bincount(samples)})"


@test("Qwen3_5MTPDrafter wraps causal LM correctly")
def test_qwen35_mtp_drafter():
    from easydel.inference.speculative import DraftStep, Qwen3_5MTPDrafter

    cfg = make_qwen35_text_config()
    model = Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    drafter = Qwen3_5MTPDrafter(model)
    factory_drafter = model.drafter(method="mtp", num_draft_tokens=3)
    assert isinstance(factory_drafter, Qwen3_5MTPDrafter), "model.drafter(method='mtp') should build MTP drafter"
    assert factory_drafter.num_draft_tokens == 3, "model.drafter should forward num_draft_tokens"
    drafter.reset(batch_size=1)
    B, S, H = 1, 4, cfg.hidden_size
    hidden = jax.random.normal(jax.random.PRNGKey(0), (B, S, H))
    input_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    step = drafter.draft(input_ids=input_ids, target_hidden_states=hidden)
    assert isinstance(step, DraftStep), f"Expected DraftStep, got {type(step)}"
    assert_shape(step.token_ids, (B,), "drafter.token_ids")
    assert step.log_probs is None, "greedy drafter should skip log_probs"
    assert step.full_log_probs is None, "greedy drafter should skip full_log_probs"

    step = drafter.draft(
        input_ids=input_ids,
        target_hidden_states=hidden,
        return_full_log_probs=True,
    )
    assert_shape(step.log_probs, (B,), "drafter.log_probs")
    assert step.full_log_probs is not None, "full_log_probs should be provided by MTP drafter"
    assert_shape(step.full_log_probs, (B, cfg.vocab_size), "drafter.full_log_probs")


@test("Qwen3_5MTPDrafter raises on missing MTP head")
def test_qwen35_drafter_without_mtp_raises():
    from easydel.inference.speculative import Qwen3_5MTPDrafter

    cfg = make_qwen35_text_config(mtp_layers=0)
    model = Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    try:
        Qwen3_5MTPDrafter(model)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when wrapping model without MTP head")


def _qwen35_2b_snap():
    p = "/dev/shm/easydel_ckpts/hf/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
    return p if os.path.isdir(p) else None


def _gemma4_e4b_asst_snap():
    p = "/dev/shm/easydel_ckpts/hf/models--google--gemma-4-E4B-it-assistant/snapshots/4a5c666f89be588c72e0b3a9b49c118513cedff6"
    return p if os.path.isdir(p) else None


@test("REAL: Qwen3.5-2B mtp.* tensors load into Qwen3_5MTPHead and forward runs")
def test_qwen35_real_mtp_load_and_forward():
    snap = _qwen35_2b_snap()
    if snap is None:
        print("       (skipped: no /dev/shm checkpoint)")
        return
    import glob

    from safetensors import safe_open

    cfg = Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        max_position_embeddings=262144,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
        mtp_num_hidden_layers=1,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
    )
    head = Qwen3_5MTPHead(config=cfg, rngs=spx.Rngs(0), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    hf = {}
    for sf in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(sf, framework="numpy") as fp:
            for k in fp.keys():
                if k.startswith("mtp."):
                    hf[k] = fp.get_tensor(k)

    state = spx.tree_state(head)
    flat_paths = jax.tree_util.tree_flatten_with_path(state)[0]

    def path_to_hf(path):
        parts = []
        for p in path:
            if hasattr(p, "key"):
                parts.append(str(p.key))
            elif hasattr(p, "idx"):
                parts.append(str(p.idx))
            else:
                parts.append(str(p))
        if parts and parts[0] == "parameters":
            parts = parts[1:]
        return "mtp." + ".".join(parts)

    new_leaves = []
    for path, leaf in flat_paths:
        name = path_to_hf(path)
        if name in hf:
            v = hf[name]
        elif name.endswith("gate_up_proj.weight"):
            prefix = name[: -len("gate_up_proj.weight")]
            gate_name = f"{prefix}gate_proj.weight"
            up_name = f"{prefix}up_proj.weight"
            if gate_name in hf and up_name in hf:
                v = np.concatenate([hf[gate_name], hf[up_name]], axis=0)
            else:
                raise AssertionError(f"missing HF tensor: {name}")
        elif name.endswith("qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            q_name = f"{prefix}q_proj.weight"
            k_name = f"{prefix}k_proj.weight"
            v_name = f"{prefix}v_proj.weight"
            if q_name in hf and k_name in hf and v_name in hf:
                v = np.concatenate([hf[q_name], hf[k_name], hf[v_name]], axis=0)
            else:
                raise AssertionError(f"missing HF tensor: {name}")
        else:
            raise AssertionError(f"missing HF tensor: {name}")
        if v.ndim == 2 and v.shape != tuple(leaf.shape):
            v = v.T
        if tuple(v.shape) != tuple(leaf.shape):
            raise AssertionError(f"shape mismatch after transpose: {name} hf={v.shape} esl={tuple(leaf.shape)}")
        new_leaves.append(jnp.asarray(v, dtype=jnp.bfloat16))
    treedef = jax.tree_util.tree_structure(state)
    new_state = jax.tree_util.tree_unflatten(treedef, new_leaves)
    spx.update(head, new_state)

    B, S, H = 1, 4, cfg.hidden_size
    prev = jax.random.normal(jax.random.PRNGKey(0), (B, S, H)).astype(jnp.bfloat16)
    nxt = jax.random.normal(jax.random.PRNGKey(1), (B, S, H)).astype(jnp.bfloat16)
    out = head(prev_hidden_states=prev, next_token_embeds=nxt)
    assert_shape(out.last_hidden_state, (B, S, H), "real-mtp.last_hidden_state")
    arr = np.asarray(out.last_hidden_state.astype(jnp.float32))
    assert not np.any(np.isnan(arr)), "Real-weights forward produced NaN"
    assert not np.any(np.isinf(arr)), "Real-weights forward produced Inf"
    assert float(np.abs(arr).mean()) > 1e-4, f"Output magnitude suspiciously small: {float(np.abs(arr).mean())}"
    del head, hf, new_leaves, state, new_state
    gc.collect()


@test("REAL: Gemma4-E4B-it-assistant loads into Gemma4AssistantForCausalLM and forward runs")
def test_gemma4_real_assistant_load_and_forward():
    snap = _gemma4_e4b_asst_snap()
    if snap is None:
        print("       (skipped: no /dev/shm checkpoint)")
        return
    import json

    from safetensors import safe_open

    cfg_json = json.load(open(os.path.join(snap, "config.json")))
    text_cfg = Gemma4AssistantTextConfig(
        vocab_size=cfg_json["text_config"]["vocab_size"],
        hidden_size=cfg_json["text_config"]["hidden_size"],
        intermediate_size=cfg_json["text_config"]["intermediate_size"],
        num_hidden_layers=cfg_json["text_config"]["num_hidden_layers"],
        num_attention_heads=cfg_json["text_config"]["num_attention_heads"],
        num_key_value_heads=cfg_json["text_config"]["num_key_value_heads"],
        head_dim=cfg_json["text_config"]["head_dim"],
        global_head_dim=cfg_json["text_config"].get("global_head_dim", cfg_json["text_config"]["head_dim"]),
        layer_types=cfg_json["text_config"]["layer_types"],
        sliding_window=cfg_json["text_config"]["sliding_window"],
        max_position_embeddings=cfg_json["text_config"]["max_position_embeddings"],
        rms_norm_eps=cfg_json["text_config"]["rms_norm_eps"],
        tie_word_embeddings=cfg_json["text_config"]["tie_word_embeddings"],
    )
    cfg = Gemma4AssistantConfig(
        text_config=text_cfg,
        backbone_hidden_size=cfg_json["backbone_hidden_size"],
        num_centroids=cfg_json["num_centroids"],
        centroid_intermediate_top_k=cfg_json["centroid_intermediate_top_k"],
        use_ordered_embeddings=cfg_json["use_ordered_embeddings"],
        tie_word_embeddings=cfg_json["tie_word_embeddings"],
    )
    model = Gemma4AssistantForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    sf = os.path.join(snap, "model.safetensors")
    hf = {}
    with safe_open(sf, framework="numpy") as fp:
        for k in fp.keys():
            hf[k] = fp.get_tensor(k)

    state = spx.tree_state(model)
    flat_paths = jax.tree_util.tree_flatten_with_path(state)[0]

    def path_to_str(path):
        parts = []
        for p in path:
            if hasattr(p, "key"):
                parts.append(str(p.key))
            elif hasattr(p, "idx"):
                parts.append(str(p.idx))
            else:
                parts.append(str(p))
        if parts and parts[0] == "parameters":
            parts = parts[1:]
        return ".".join(parts)

    new_leaves = []
    matched_keys = set()
    for path, leaf in flat_paths:
        name = path_to_str(path)
        if name in hf:
            v = hf[name]
            if v.ndim == 2 and v.shape != tuple(leaf.shape):
                v = v.T
            if v.shape != tuple(leaf.shape):
                raise AssertionError(f"shape mismatch: {name} hf={v.shape} esl={tuple(leaf.shape)}")
            if leaf.dtype == jnp.int32 or leaf.dtype == jnp.int64:
                new_leaves.append(jnp.asarray(v, dtype=jnp.int32))
            else:
                new_leaves.append(jnp.asarray(v, dtype=jnp.bfloat16))
            matched_keys.add(name)
        else:
            new_leaves.append(leaf)
    treedef = jax.tree_util.tree_structure(state)
    new_state = jax.tree_util.tree_unflatten(treedef, new_leaves)
    spx.update(model, new_state)

    unmatched = set(hf) - matched_keys
    if unmatched:
        raise AssertionError(f"unmatched HF tensors: {sorted(unmatched)[:5]}")

    B, S = 1, 4
    BH = cfg.backbone_hidden_size
    backbone_h = jax.random.normal(jax.random.PRNGKey(0), (B, S, BH)).astype(jnp.bfloat16)
    target_e = jax.random.normal(jax.random.PRNGKey(1), (B, S, BH)).astype(jnp.bfloat16)
    out = model(backbone_hidden_states=backbone_h, target_token_embeds=target_e, return_dense_logits=False)
    assert_shape(out.last_hidden_state, (B, S, text_cfg.hidden_size), "real-asst.last_hidden_state")
    assert_shape(out.backbone_hidden_state, (B, S, BH), "real-asst.backbone_hidden_state")
    arr = np.asarray(out.last_hidden_state.astype(jnp.float32))
    assert not np.any(np.isnan(arr)), "Real-Gemma4 forward produced NaN"
    ids = np.asarray(out.top_token_ids)
    assert ids.min() >= 0 and ids.max() < text_cfg.vocab_size
    del model, hf, new_leaves, state, new_state
    gc.collect()


@test("PERF: Qwen3.5 MTP head JIT-compile + per-call latency (S=128, S=512, S=2048)")
def perf_qwen35_mtp_latency():
    """Profile MTP head at varying sequence lengths to characterize
    scaling on the current device. Reports per-token us so the cost
    can be compared to the main model's per-token decode cost (which
    is in the few-ms range for the 27B target)."""
    cfg = make_qwen35_text_config(hidden=512, vocab=1024, mtp_layers=1)
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 2
    cfg.head_dim = 64
    cfg.intermediate_size = 1024
    cfg.max_position_embeddings = 4096
    head = Qwen3_5MTPHead(config=cfg, rngs=spx.Rngs(0), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    timings = []
    for S in (128, 512, 2048):
        B, H = 1, cfg.hidden_size
        prev = jax.random.normal(jax.random.PRNGKey(S), (B, S, H)).astype(jnp.bfloat16)
        nxt = jax.random.normal(jax.random.PRNGKey(S + 1), (B, S, H)).astype(jnp.bfloat16)

        @jax.jit
        def run(p, n):
            return head(prev_hidden_states=p, next_token_embeds=n).last_hidden_state

        t0 = time.time()
        _ = run(prev, nxt).block_until_ready()
        compile_ms = (time.time() - t0) * 1000

        ITERS = 20
        t0 = time.time()
        for _ in range(ITERS):
            _ = run(prev, nxt).block_until_ready()
        steady_ms = (time.time() - t0) * 1000 / ITERS
        per_tok_us = steady_ms * 1000 / S
        timings.append((S, compile_ms, steady_ms, per_tok_us))
        print(f"       S={S:4d}  compile={compile_ms:.0f}ms  steady={steady_ms:.3f}ms  per-token={per_tok_us:.2f}us")
    for S, _compile_ms, steady_ms, _per_tok_us in timings:
        assert steady_ms < 5000, f"MTP head too slow at S={S}: {steady_ms} ms"


@test("PERF: Gemma4 centroid head scales as B*S grows (1x32 → 4x256)")
def perf_centroid_head_scaling():
    """Show the centroid head's wallclock advantage growing with B*S."""
    cfg = make_gemma4_assistant_config(hidden=256, vocab=262144, num_centroids=2048)
    head = ed.Gemma4AssistantCentroidHead(config=cfg, rngs=spx.Rngs(0), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    V = cfg.text_config.vocab_size
    H = cfg.text_config.hidden_size
    embed_w = jax.random.normal(jax.random.PRNGKey(1), (V, H)).astype(jnp.bfloat16)

    @jax.jit
    def centroid_only(h, e):
        tl, ti, _ = head(h, e, return_dense_logits=False)
        return tl, ti

    @jax.jit
    def full_only(h, e):
        return jnp.einsum("bsh,vh->bsv", h.astype(jnp.float32), e.astype(jnp.float32))

    results = []
    for B, S in [(1, 32), (1, 128), (4, 128), (4, 256)]:
        hidden = jax.random.normal(jax.random.PRNGKey(B * 1000 + S), (B, S, H)).astype(jnp.bfloat16)
        a, _b = centroid_only(hidden, embed_w)
        a.block_until_ready()
        f = full_only(hidden, embed_w)
        f.block_until_ready()
        ITERS = 20
        t0 = time.time()
        for _ in range(ITERS):
            a, _b = centroid_only(hidden, embed_w)
            a.block_until_ready()
        c_ms = (time.time() - t0) * 1000 / ITERS
        t0 = time.time()
        for _ in range(ITERS):
            f = full_only(hidden, embed_w)
            f.block_until_ready()
        f_ms = (time.time() - t0) * 1000 / ITERS
        sx = f_ms / c_ms
        results.append((B, S, c_ms, f_ms, sx))
    for B, S, c, f, sx in results:
        print(f"       B={B:2d} S={S:3d}  centroid={c:.3f}ms  full={f:.3f}ms  speedup={sx:.2f}x")


@test("PERF: Gemma4 centroid head vs full-softmax FLOP savings")
def perf_gemma4_centroid_vs_full():
    cfg = make_gemma4_assistant_config(hidden=256, vocab=262144, num_centroids=2048)
    head = Gemma4AssistantCentroidHead(config=cfg, rngs=spx.Rngs(0), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    B, S, H = 1, 32, cfg.text_config.hidden_size
    V = cfg.text_config.vocab_size
    hidden = jax.random.normal(jax.random.PRNGKey(0), (B, S, H)).astype(jnp.bfloat16)
    embed_w = jax.random.normal(jax.random.PRNGKey(1), (V, H)).astype(jnp.bfloat16)

    @jax.jit
    def centroid_path(h, e):
        top_logits, top_ids, _ = head(h, e, return_dense_logits=False)
        return top_logits, top_ids

    @jax.jit
    def full_path(h, e):
        return jnp.einsum("bsh,vh->bsv", h.astype(jnp.float32), e.astype(jnp.float32))

    a, _b = centroid_path(hidden, embed_w)
    a.block_until_ready()
    full_logits = full_path(hidden, embed_w)
    full_logits.block_until_ready()

    ITERS = 20
    t0 = time.time()
    for _ in range(ITERS):
        a, _b = centroid_path(hidden, embed_w)
        a.block_until_ready()
    centroid_ms = (time.time() - t0) * 1000 / ITERS

    t0 = time.time()
    for _ in range(ITERS):
        full_logits = full_path(hidden, embed_w)
        full_logits.block_until_ready()
    full_ms = (time.time() - t0) * 1000 / ITERS

    centroid_flops = head.num_centroids + head.top_k * head.tokens_per_centroid
    speedup_theory = V / centroid_flops
    speedup_actual = full_ms / centroid_ms
    device_kind = jax.devices()[0].platform
    print(
        f"       device={device_kind}  centroid={centroid_ms:.3f}ms  full={full_ms:.3f}ms  "
        f"actual_speedup={speedup_actual:.2f}x  theoretical={speedup_theory:.1f}x"
    )
    if device_kind == "cpu":
        assert centroid_ms < full_ms * 5.0, (
            f"centroid head wallclock {centroid_ms}ms is much slower than full {full_ms}ms on CPU"
        )


@test("REAL: Qwen3.5 MTP forward output is non-degenerate (varies per position + per batch)")
def test_qwen35_real_mtp_output_nondegenerate():
    """Same load path as test_qwen35_real_mtp_load_and_forward but with
    correctness checks on the output (no NaN, finite, varies)."""
    snap = _qwen35_2b_snap()
    if snap is None:
        print("       (skipped: no /dev/shm checkpoint)")
        return
    import glob

    from safetensors import safe_open

    cfg = Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        max_position_embeddings=262144,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
        mtp_num_hidden_layers=1,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
    )
    head = Qwen3_5MTPHead(config=cfg, rngs=spx.Rngs(0), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    hf = {}
    for sf in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(sf, framework="numpy") as fp:
            for k in fp.keys():
                if k.startswith("mtp."):
                    hf[k] = fp.get_tensor(k)
    state = spx.tree_state(head)
    flat_paths = jax.tree_util.tree_flatten_with_path(state)[0]

    def path_to_hf(path):
        parts = []
        for p in path:
            if hasattr(p, "key"):
                parts.append(str(p.key))
            elif hasattr(p, "idx"):
                parts.append(str(p.idx))
            else:
                parts.append(str(p))
        if parts and parts[0] == "parameters":
            parts = parts[1:]
        return "mtp." + ".".join(parts)

    new_leaves = []
    for path, leaf in flat_paths:
        name = path_to_hf(path)
        if name in hf:
            v = hf[name]
        elif name.endswith("gate_up_proj.weight"):
            prefix = name[: -len("gate_up_proj.weight")]
            gate_name = f"{prefix}gate_proj.weight"
            up_name = f"{prefix}up_proj.weight"
            if gate_name in hf and up_name in hf:
                v = np.concatenate([hf[gate_name], hf[up_name]], axis=0)
            else:
                raise KeyError(name)
        elif name.endswith("qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            q_name = f"{prefix}q_proj.weight"
            k_name = f"{prefix}k_proj.weight"
            v_name = f"{prefix}v_proj.weight"
            if q_name in hf and k_name in hf and v_name in hf:
                v = np.concatenate([hf[q_name], hf[k_name], hf[v_name]], axis=0)
            else:
                raise KeyError(name)
        else:
            raise KeyError(name)
        if v.ndim == 2 and v.shape != tuple(leaf.shape):
            v = v.T
        new_leaves.append(jnp.asarray(v, dtype=jnp.bfloat16))
    new_state = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(state), new_leaves)
    spx.update(head, new_state)

    _B, S, H = 2, 8, cfg.hidden_size
    prev_a = jax.random.normal(jax.random.PRNGKey(0), (1, S, H)).astype(jnp.bfloat16)
    prev_b = jax.random.normal(jax.random.PRNGKey(1), (1, S, H)).astype(jnp.bfloat16)
    nxt_a = jax.random.normal(jax.random.PRNGKey(2), (1, S, H)).astype(jnp.bfloat16)
    nxt_b = jax.random.normal(jax.random.PRNGKey(3), (1, S, H)).astype(jnp.bfloat16)
    prev = jnp.concatenate([prev_a, prev_b], axis=0)
    nxt = jnp.concatenate([nxt_a, nxt_b], axis=0)
    out = head(prev_hidden_states=prev, next_token_embeds=nxt)
    arr = np.asarray(out.last_hidden_state.astype(jnp.float32))
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))
    diff_across_batch = float(np.abs(arr[0] - arr[1]).mean())
    assert diff_across_batch > 1e-3, f"output identical across distinct batch items: {diff_across_batch}"
    diff_across_seq = float(np.abs(arr[0, 0] - arr[0, S - 1]).mean())
    assert diff_across_seq > 1e-3, f"output identical across distinct positions: {diff_across_seq}"
    del head, hf, new_leaves, state, new_state
    gc.collect()


@test("E2E: speculative-decode loop produces accepted+resampled tokens with no NaN")
def test_speculative_decode_e2e():
    """Synthetic 4-step speculative-decode loop: drafter proposes,
    target verifies (with random tlp); resample on reject. Verifies the
    full chain produces valid tokens."""
    from easydel.inference.speculative import accept_or_reject, resample_rejected

    V = 64
    rng = jax.random.PRNGKey(123)
    accepted_total = 0
    rejected_total = 0
    for step in range(4):
        rng, key1, key2, key3 = jax.random.split(rng, 4)
        drf_logits = jax.random.normal(key1, (4, V))
        drf_lp_full = jax.nn.log_softmax(drf_logits, axis=-1)
        draft_tok = jnp.argmax(drf_logits, axis=-1)
        draft_tok_lp = jnp.take_along_axis(drf_lp_full, draft_tok[:, None], axis=-1).squeeze(-1)
        tgt_logits = jax.random.normal(key2, (4, V))
        tgt_lp_full = jax.nn.log_softmax(tgt_logits, axis=-1)
        tgt_tok_lp = jnp.take_along_axis(tgt_lp_full, draft_tok[:, None], axis=-1).squeeze(-1)
        accept = accept_or_reject(draft_tok_lp, tgt_tok_lp, key3)
        rng, key4 = jax.random.split(rng)
        replacement = resample_rejected(tgt_lp_full, drf_lp_full, key4)
        final_tokens = jnp.where(accept.astype(jnp.bool_), draft_tok, replacement)
        accepted_total += int(jnp.sum(accept))
        rejected_total += int(4 - jnp.sum(accept))
        ft = np.asarray(final_tokens)
        assert ft.min() >= 0 and ft.max() < V, f"step {step}: invalid token IDs"
    assert accepted_total + rejected_total == 16
    print(f"       4-step spec-decode: accepted={accepted_total} rejected={rejected_total}")


ALL_TESTS = [
    test_qwen35_mtp_forward_shape,
    test_qwen35_mtp_single_token,
    test_qwen35_for_causal_lm_mtp_e2e,
    test_qwen35_no_mtp_when_disabled,
    test_mtp_forward_populates_aux_loss,
    test_mtp_head_receives_gradients,
    test_compute_mtp_loss_math,
    test_compute_mtp_loss_attention_mask,
    test_gemma4_centroid_head,
    test_gemma4_centroid_dense_scatter,
    test_gemma4_assistant_forward_shape,
    test_accept_or_reject_dominant_target,
    test_accept_or_reject_dominated_target,
    test_resample_rejected,
    test_qwen35_mtp_drafter,
    test_qwen35_drafter_without_mtp_raises,
    test_qwen35_real_mtp_load_and_forward,
    test_gemma4_real_assistant_load_and_forward,
    test_qwen35_real_mtp_output_nondegenerate,
    test_speculative_decode_e2e,
    perf_qwen35_mtp_latency,
    perf_gemma4_centroid_vs_full,
    perf_centroid_head_scaling,
]


if __name__ == "__main__":
    print("=" * 80)
    print(f"Running {len(ALL_TESTS)} tests")
    print("=" * 80)
    for tfn in ALL_TESTS:
        tfn()
    print()
    print("=" * 80)
    n_pass = sum(1 for s, *_ in _RESULTS if s == "PASS")
    n_fail = sum(1 for s, *_ in _RESULTS if s == "FAIL")
    print(f"Summary: {n_pass} passed, {n_fail} failed (total: {len(_RESULTS)})")
    print("=" * 80)
    sys.exit(0 if n_fail == 0 else 1)
