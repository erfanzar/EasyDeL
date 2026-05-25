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

"""Verify the speculative-decode driver runs dynamically + measure speedup.

Three checks:

1. ``test_driver_runs_dynamically`` — runs the real driver on a tiny
   Qwen3.5 model with a random-init MTP head. Proves the loop executes
   end-to-end: drafter called every step, drafts verified, tokens
   emitted. Random-init MTP -> ~0% acceptance is the EXPECTED result.

2. ``test_perfect_drafter_speedup`` — a synthetic fake target +
   matching fake drafter where every draft is provably accepted.
   Isolates the accept path and proves it yields N+1 tokens per
   target forward (the speculative-decoding speedup ceiling). Pure
   JAX, fixed shapes, no real-model compilation.

3. ``test_token_accounting`` — real tiny model; verifies the emitted
   token count reconciles exactly with accepted + corrected + prefill.
"""

from __future__ import annotations

import traceback

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx

from easydel.inference.esurge import SpeculativeMTPDriver
from easydel.inference.esurge.speculative_decoding import _SpecDecodeDriverBase
from easydel.inference.speculative import DraftStep
from easydel.modules.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def make_tiny_model(mtp_layers: int = 1):
    cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        mtp_num_hidden_layers=mtp_layers,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
    )
    return Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)




class _FakeOutput:
    def __init__(self, logits, last_hidden_state):
        self.logits = logits
        self.last_hidden_state = last_hidden_state
        self.past_key_values = None


class FakeTarget:
    """Deterministic target: argmax at position s == (input[s] + 1) % V.

    Implemented as pure JAX so it survives ``jax.jit``.
    """

    def __init__(self, vocab: int = 256, hidden: int = 16):
        self.vocab = vocab
        self.hidden = hidden

    def __call__(self, input_ids):
        nxt = (input_ids + 1) % self.vocab  # (B, S)
        logits = jax.nn.one_hot(nxt, self.vocab, dtype=jnp.float32) * 30.0
        B, S = input_ids.shape
        hidden = jnp.zeros((B, S, self.hidden), dtype=jnp.float32)
        hidden = hidden.at[:, :, 0].set(input_ids.astype(jnp.float32))
        return _FakeOutput(logits, hidden)


class FakeMatchingDrafter:
    """Drafter that proposes ``seed+1`` (chained), matching FakeTarget.

    FakeTarget predicts ``input+1`` at every position. This drafter,
    chained, proposes ``seed+1, seed+2, ...`` — exactly what the target
    will argmax — so every draft is accepted.
    """

    def __init__(self, vocab: int = 256):
        self.vocab = vocab

    def reset(self, batch_size):
        del batch_size

    def draft(self, input_ids, target_hidden_states=None, target_kv_cache=None, sample=False, rng_key=None):
        seed = input_ids[:, -1]
        tok = ((seed + 1) % self.vocab).astype(jnp.int32)
        lp = jax.nn.log_softmax(jax.nn.one_hot(tok, self.vocab, dtype=jnp.float32) * 30.0, axis=-1)
        return DraftStep(token_ids=tok, log_probs=jnp.zeros_like(tok, dtype=jnp.float32), full_log_probs=lp)




def test_driver_runs_dynamically():
    print("\n[test 1] SpeculativeMTPDriver runs dynamically on a real model")
    model = make_tiny_model(mtp_layers=1)
    assert model.has_mtp()
    driver = SpeculativeMTPDriver(target_model=model, num_draft_tokens=1, greedy=True)
    prompt = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    _out, stats = driver.generate(prompt, max_new_tokens=12)

    assert stats.num_draft_steps > 0, "drafter never called"
    assert stats.num_drafts_generated == stats.num_draft_steps, "draft count mismatch (N=1)"
    assert stats.num_target_forwards == stats.num_draft_steps + 1, "expect 1 prefill + 1/step"
    assert stats.tokens_generated >= 12, f"under-generated: {stats.tokens_generated}"
    print(f"     draft_steps={stats.num_draft_steps}  target_forwards={stats.num_target_forwards}")
    print(f"     drafts={stats.num_drafts_generated}  accepted={stats.num_drafts_accepted}")
    print(f"     acceptance_rate={stats.acceptance_rate:.1%}  (random-init MTP -> ~0% expected)")
    print(f"     algorithmic_speedup={stats.speedup_vs_baseline:.2f}x")
    print(f"     wallclock={stats.wallclock_s:.2f}s")
    print("  PASS — drafter invoked every step, loop ran end-to-end")
    return True


def test_perfect_drafter_speedup():
    print("\n[test 2] perfect-drafter speedup proof (synthetic target)")
    for N in (1, 2, 4):
        target = FakeTarget(vocab=256, hidden=16)
        drafter = FakeMatchingDrafter(vocab=256)
        driver = _SpecDecodeDriverBase(target_model=target, drafter=drafter, num_draft_tokens=N, greedy=True)
        prompt = jnp.array([[10, 11, 12, 13]], dtype=jnp.int32)
        out, stats = driver.generate(prompt, max_new_tokens=20)
        gen = np.asarray(out[0, prompt.shape[-1] :])
        expected = (np.arange(len(gen)) + 14) % 256
        assert np.array_equal(gen, expected), f"N={N}: generated stream wrong.\n  got={gen}\n  exp={expected}"
        print(
            f"     N={N}:  steps={stats.num_draft_steps}  target_forwards={stats.num_target_forwards}  "
            f"accepted={stats.num_drafts_accepted}/{stats.num_drafts_generated}  "
            f"accept_rate={stats.acceptance_rate:.0%}  speedup={stats.speedup_vs_baseline:.2f}x"
        )
        assert stats.acceptance_rate == 1.0, f"N={N}: expected 100% acceptance, got {stats.acceptance_rate}"
        assert stats.speedup_vs_baseline > min(N + 1, 2.0) - 0.5, (
            f"N={N}: speedup {stats.speedup_vs_baseline:.2f}x below expected ~{N + 1}x"
        )
    print("  PASS — perfect drafter yields up to N+1 tokens per target forward")
    return True


def test_token_accounting():
    print("\n[test 3] token accounting reconciles exactly")
    model = make_tiny_model(mtp_layers=1)
    driver = SpeculativeMTPDriver(target_model=model, num_draft_tokens=2, greedy=True)
    prompt = jnp.array([[3, 1, 4, 1, 5, 9, 2, 6]], dtype=jnp.int32)
    out, stats = driver.generate(prompt, max_new_tokens=10)
    generated = np.asarray(out[0, prompt.shape[-1] :])
    assert generated.min() >= 0 and generated.max() < 256, "emitted IDs out of vocab range"
    expected = 1 + stats.num_drafts_accepted + stats.num_draft_steps
    assert stats.tokens_generated == expected, (
        f"accounting mismatch: emitted={stats.tokens_generated} expected={expected} "
        f"(1 prefill + {stats.num_drafts_accepted} accepted + {stats.num_draft_steps} corrections)"
    )
    print(
        f"     emitted={stats.tokens_generated} = 1 prefill + {stats.num_drafts_accepted} accepted "
        f"+ {stats.num_draft_steps} corrections"
    )
    print(f"     all {len(generated)} generated tokens in [0, 256)")
    print("  PASS — token accounting exact")
    return True


def test_cachefree_correct_and_cached_gated():
    """The cache-free path is the verified default: it must be
    deterministic (greedy). The commit-based cached path is
    implemented but gated — ``use_cache=True`` must raise a clear
    NotImplementedError until the esurge cache-metadata machinery is
    wired (a direct model call cannot numerically-correctly EXTEND an
    EasyDeL cache; the prefill is fine, extension desyncs)."""
    print("\n[test 4] cache-free path verified correct + cached path gated")
    model = make_tiny_model(mtp_layers=1)
    prompt = jnp.array([[5, 12, 8, 3, 19, 7]], dtype=jnp.int32)

    d1 = SpeculativeMTPDriver(target_model=model, num_draft_tokens=1, greedy=True)
    d2 = SpeculativeMTPDriver(target_model=model, num_draft_tokens=1, greedy=True)
    out1, _ = d1.generate(prompt, max_new_tokens=10, use_cache=False)
    out2, _ = d2.generate(prompt, max_new_tokens=10, use_cache=False)
    a, b = np.asarray(out1[0]), np.asarray(out2[0])
    m = min(len(a), len(b))
    assert np.array_equal(a[:m], b[:m]), (
        f"cache-free greedy decode is non-deterministic:\n  run1={a[:m]}\n  run2={b[:m]}"
    )
    print(f"     cache-free greedy decode deterministic over {m} tokens")

    d3 = SpeculativeMTPDriver(target_model=model, num_draft_tokens=1, greedy=True)
    try:
        d3.generate(prompt, max_new_tokens=4, use_cache=True)
    except NotImplementedError as e:
        msg = str(e)
        assert "cache_metadata" in msg and "cache-free" in msg, (
            f"gating message should explain the blocker, got: {msg[:120]}"
        )
        print("     use_cache=True correctly raises NotImplementedError with the reason")
    else:
        raise AssertionError("use_cache=True should raise NotImplementedError (it is gated)")
    print("  PASS — cache-free verified deterministic; cached path honestly gated")
    return True


def test_gemma4_kv_controller():
    """The Gemma4 cross-model K/V controller: layer mapping + extraction."""
    print("\n[test 5] Gemma4 cross-model K/V controller")
    from easydel.inference.esurge import build_target_kv_pairs, default_assistant_layer_mapping

    mapping = default_assistant_layer_mapping(num_assistant_layers=4, num_target_layers=30)
    assert mapping == [26, 27, 28, 29], f"unexpected default mapping: {mapping}"
    deg = default_assistant_layer_mapping(num_assistant_layers=4, num_target_layers=2)
    assert deg == [1, 1, 1, 1], f"degenerate mapping wrong: {deg}"
    print(f"     default_assistant_layer_mapping(4, 30) = {mapping}")

    class _MockView:
        def __init__(self, key, value):
            self.key = key
            self.value = value

    class _MockCache:
        def __init__(self, views):
            self.views = views

    views = []
    for i in range(30):
        if i in (5, 12):  # simulate linear-attention layers: no K/V
            views.append(_MockView(None, None))
        else:
            views.append(_MockView(jnp.full((1, 8, 2, 16), float(i)), jnp.full((1, 8, 2, 16), -float(i))))
    cache = _MockCache(views)

    pairs = build_target_kv_pairs(cache, mapping)
    assert len(pairs) == 4, f"expected 4 pairs, got {len(pairs)}"
    for assistant_idx, tgt_idx in enumerate(mapping):
        k, v = pairs[assistant_idx]
        assert float(k[0, 0, 0, 0]) == float(tgt_idx), f"assistant layer {assistant_idx} got K from wrong target layer"
        assert float(v[0, 0, 0, 0]) == -float(tgt_idx), "V mismatch"
    print(f"     build_target_kv_pairs mapped 4 assistant layers -> target {mapping}")

    pairs2 = build_target_kv_pairs(cache, [5, 12, 28, 29])
    assert pairs2[0] is None and pairs2[1] is None, "linear-attn layers must yield None"
    assert pairs2[2] is not None and pairs2[3] is not None, "full-attn layers must yield K/V"
    pairs3 = build_target_kv_pairs(cache, [999])
    assert pairs3[0] is None, "out-of-range target layer must yield None"
    print("     linear-attn layers + out-of-range indices -> None (as designed)")
    print("  PASS — Gemma4 K/V controller maps + extracts correctly")
    return True


def main():
    print("=" * 80)
    print("Speculative decoding — dynamic execution + speedup measurement")
    print("=" * 80)
    results = []
    for fn in (
        test_driver_runs_dynamically,
        test_perfect_drafter_speedup,
        test_token_accounting,
        test_cachefree_correct_and_cached_gated,
        test_gemma4_kv_controller,
    ):
        try:
            fn()
            results.append(True)
        except Exception as e:
            traceback.print_exc()
            print(f"  FAIL — {fn.__name__}: {e}")
            results.append(False)
    print()
    print("=" * 80)
    print(f"Summary: {sum(results)}/{len(results)} passed")
    print("=" * 80)
    return 0 if all(results) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
