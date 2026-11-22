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

"""Distribution-correctness tests for eSurge sampled speculative decoding.

The runner-native sampled spec-decode path runs the rejection-sampling
correction fully on device in
``DrafterSpeculation.verify_sampled_window`` (only the accepted count and the
corrected/bonus token id cross to host). The standard speculative-decoding
guarantee (Leviathan et al., 2023) is that the emitted token stream is
distributed exactly as if it had been sampled token-by-token from the
target model's *sampling* distribution (i.e. the target logits after the
request's temperature / top-k / top-p / min-p filter).

These tests verify that guarantee empirically for the position-0 emission of a
verify window: draw the draft tokens from the (filtered) drafter distribution
exactly as the runner does, run the on-device rejection sampler many times, and
assert the histogram of the first emitted token matches the filtered target
distribution — both with no filtering and under nucleus (top-p) filtering.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.inference.sampling_params import SamplingParams


class _StubExec:
    """Minimal executor exposing the mutable RNG state ``rng_split`` advances."""

    def __init__(self, seed: int = 0):
        self.rng_key = jax.random.PRNGKey(seed)


class _StubRunner:
    """Minimal runner back-reference: ``verify_sampled_window`` only needs the RNG."""

    def __init__(self, seed: int = 0):
        self.executor_manager = _StubExec(seed)


class _StubReqState:
    """Carries only ``sampling_params`` — all the strategy reads for sampling."""

    def __init__(self, sampling_params):
        self.sampling_params = sampling_params


def _make_spec(seed: int = 0) -> DrafterSpeculation:
    return DrafterSpeculation(runner=_StubRunner(seed), drafter=None, num_draft_tokens=2)


def _filtered_probs(logits: jax.Array, sp: SamplingParams) -> np.ndarray:
    """Reference target *sampling* distribution: the request filter, then softmax.

    Uses the exact same filter the on-device verifier applies so the reference
    is the distribution the guarantee promises to reproduce.
    """
    filtered = DrafterSpeculation.filter_logits_for_sampling(
        logits[None, :],
        temperature=jnp.asarray(float(sp.temperature or 1.0), dtype=jnp.float32),
        top_p=jnp.asarray(float(sp.top_p or 1.0), dtype=jnp.float32),
        top_k=jnp.asarray(int(sp.top_k or 0), dtype=jnp.int32),
        min_p=jnp.asarray(float(sp.min_p or 0.0), dtype=jnp.float32),
    )
    return np.asarray(jax.nn.softmax(filtered[0], axis=-1))


def _empirical_position0(
    *,
    target_logits: jax.Array,
    draft_logits_pos0: jax.Array,
    sp: SamplingParams,
    num_draft_tokens: int,
    num_trials: int,
    seed: int,
) -> np.ndarray:
    """Run the on-device rejection sampler ``num_trials`` times; histogram token 0.

    Mirrors the runner: the drafter distribution is the request-filtered draft
    distribution, draft tokens are sampled from it, and the stored
    ``draft_log_probs`` are its filtered log-probs. ``target_logits`` are RAW
    (the verifier filters them internally). Returns the empirical distribution
    of the FIRST emitted token of the window.
    """
    vocab = int(target_logits.shape[-1])
    spec = _make_spec(seed)

    # Filtered drafter distribution q (exactly what draft_next samples from and
    # stores), replicated across the K draft positions for a clean fixed setup.
    draft_filtered = DrafterSpeculation.filter_logits_for_sampling(
        draft_logits_pos0[None, :],
        temperature=jnp.asarray(float(sp.temperature or 1.0), dtype=jnp.float32),
        top_p=jnp.asarray(float(sp.top_p or 1.0), dtype=jnp.float32),
        top_k=jnp.asarray(int(sp.top_k or 0), dtype=jnp.int32),
        min_p=jnp.asarray(float(sp.min_p or 0.0), dtype=jnp.float32),
    )[0]
    draft_log_probs = jnp.broadcast_to(
        jax.nn.log_softmax(draft_filtered, axis=-1)[None, :],
        (num_draft_tokens, vocab),
    )

    # Pre-sample all draft tokens from q in one batched device call.
    draft_key = jax.random.PRNGKey(seed + 7919)
    draft_tokens_all = np.asarray(
        jax.random.categorical(
            draft_key,
            jnp.broadcast_to(draft_filtered[None, :], (num_trials, vocab)),
            axis=-1,
        )
    ).astype(np.int64)

    counts = np.zeros(vocab, dtype=np.int64)
    for i in range(num_trials):
        d0 = int(draft_tokens_all[i])
        # Fill the remaining draft positions deterministically; they only affect
        # positions >= 1 and never the position-0 emission under test.
        draft_tokens = [d0, *([0] * (num_draft_tokens - 1))]
        accepted, corrected = spec.verify_sampled_window(
            target_logits=target_logits,
            draft_log_probs=draft_log_probs,
            draft_tokens=draft_tokens,
            req_state=_StubReqState(sp),
        )
        emitted0 = d0 if accepted >= 1 else int(corrected)
        counts[emitted0] += 1
    return counts / float(num_trials)


def test_verify_sampled_window_reproduces_target_distribution_no_filter():
    print("\n[test] sampled spec-decode reproduces target distribution (no filter)")
    # Target favors token 2; drafter favors token 5 -> exercises accept + residual.
    target_pos0 = jnp.asarray([0.2, 0.1, 2.4, 0.3, -0.5, 0.8, 0.0, 0.4], dtype=jnp.float32)
    target_bonus = jnp.asarray([0.0, 0.5, 0.1, 0.2, 1.1, 0.3, 0.7, 0.2], dtype=jnp.float32)
    target_logits = jnp.stack([target_pos0, target_bonus], axis=0)  # K=1 -> [K+1, V]
    draft_pos0 = jnp.asarray([0.1, 0.0, 0.3, 0.2, 0.1, 2.2, 0.4, 0.1], dtype=jnp.float32)

    sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0)
    ref = _filtered_probs(target_pos0, sp)  # == softmax(target_pos0)

    emp = _empirical_position0(
        target_logits=target_logits,
        draft_logits_pos0=draft_pos0,
        sp=sp,
        num_draft_tokens=1,
        num_trials=20000,
        seed=0,
    )
    max_abs = float(np.max(np.abs(emp - ref)))
    print(f"     ref={np.round(ref, 3)}")
    print(f"     emp={np.round(emp, 3)}  max_abs_err={max_abs:.4f}")
    assert max_abs < 0.02, f"emitted position-0 distribution deviates from target: {max_abs}"
    print("  PASS")


def test_verify_sampled_window_reproduces_target_distribution_topp():
    print("\n[test] sampled spec-decode reproduces FILTERED target distribution (top_p)")
    vocab = 12
    rng = np.random.default_rng(1234)
    target_pos0 = jnp.asarray(rng.normal(size=vocab) * 1.5, dtype=jnp.float32)
    target_bonus = jnp.asarray(rng.normal(size=vocab) * 1.5, dtype=jnp.float32)
    target_logits = jnp.stack([target_pos0, target_bonus], axis=0)
    # Deliberately mismatched drafter so acceptance < 1 and the residual path runs.
    draft_pos0 = jnp.asarray(rng.normal(size=vocab) * 1.5, dtype=jnp.float32)

    sp = SamplingParams(max_tokens=1, temperature=0.9, top_p=0.8, top_k=0, min_p=0.0)
    ref = _filtered_probs(target_pos0, sp)  # softmax(top_p-and-temp-filtered target)

    emp = _empirical_position0(
        target_logits=target_logits,
        draft_logits_pos0=draft_pos0,
        sp=sp,
        num_draft_tokens=1,
        num_trials=20000,
        seed=3,
    )
    max_abs = float(np.max(np.abs(emp - ref)))
    # Every token the target's nucleus assigns zero probability must never be
    # emitted (distribution support is exactly the filtered target's support).
    zero_support = np.asarray(ref) <= 0.0
    leaked = float(np.max(emp[zero_support])) if np.any(zero_support) else 0.0
    print(f"     ref={np.round(ref, 3)}")
    print(f"     emp={np.round(emp, 3)}  max_abs_err={max_abs:.4f}  leaked_mass={leaked:.4f}")
    assert leaked == 0.0, f"emitted tokens outside the filtered target support: {leaked}"
    assert max_abs < 0.02, f"emitted position-0 distribution deviates from filtered target: {max_abs}"
    print("  PASS")


def test_verify_sampled_window_position0_marginal_k2():
    print("\n[test] sampled spec-decode position-0 marginal holds for K=2")
    vocab = 10
    rng = np.random.default_rng(77)
    target_pos0 = jnp.asarray(rng.normal(size=vocab) * 1.3, dtype=jnp.float32)
    target_pos1 = jnp.asarray(rng.normal(size=vocab) * 1.3, dtype=jnp.float32)
    target_bonus = jnp.asarray(rng.normal(size=vocab) * 1.3, dtype=jnp.float32)
    target_logits = jnp.stack([target_pos0, target_pos1, target_bonus], axis=0)  # [K+1, V], K=2
    draft_pos0 = jnp.asarray(rng.normal(size=vocab) * 1.3, dtype=jnp.float32)

    sp = SamplingParams(max_tokens=2, temperature=1.0, top_p=0.9, top_k=0, min_p=0.0)
    ref = _filtered_probs(target_pos0, sp)

    emp = _empirical_position0(
        target_logits=target_logits,
        draft_logits_pos0=draft_pos0,
        sp=sp,
        num_draft_tokens=2,
        num_trials=20000,
        seed=11,
    )
    max_abs = float(np.max(np.abs(emp - ref)))
    print(f"     ref={np.round(ref, 3)}")
    print(f"     emp={np.round(emp, 3)}  max_abs_err={max_abs:.4f}")
    assert max_abs < 0.02, f"K=2 emitted position-0 distribution deviates from target: {max_abs}"
    print("  PASS")


def test_raw_logits_equivalent_to_log_softmax_under_filter():
    """The drafter now hands raw logits (not log-probs) on the resample path.

    ``Qwen3_5MTPDrafter.draft(return_full_log_probs=True, sample=False)`` returns
    the RAW last-position logits instead of ``log_softmax(last)`` to avoid a
    redundant full-vocab pass; the caller (``filter_and_sample_log_probs`` ->
    ``verify_sampled_window``) filters + ``log_softmax``-normalizes them itself.
    This asserts that substitution is numerically equivalent: for the same RNG,
    filtering raw logits vs pre-normalized log-probs yields the same sampled
    token and the same filtered probability distribution (the stored draft
    distribution used by the on-device rejection). Any masked position has
    exactly zero probability under both, so the residual math is unaffected.
    """
    print("\n[test] raw-logit resample input is equivalent to pre-normalized log-probs")
    vocab = 64
    key = jax.random.PRNGKey(2024)
    param_sets = [
        dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0),
        dict(temperature=0.7, top_p=0.9, top_k=0, min_p=0.0),
        dict(temperature=1.0, top_p=1.0, top_k=8, min_p=0.0),
        dict(temperature=1.3, top_p=0.95, top_k=20, min_p=0.05),
    ]
    for _trial in range(16):
        key, lk, sk = jax.random.split(key, 3)
        logits = jax.random.normal(lk, (1, vocab), dtype=jnp.float32) * 2.0
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        for params in param_sets:
            targs = dict(
                temperature=jnp.asarray(params["temperature"], dtype=jnp.float32),
                top_p=jnp.asarray(params["top_p"], dtype=jnp.float32),
                top_k=jnp.asarray(params["top_k"], dtype=jnp.int32),
                min_p=jnp.asarray(params["min_p"], dtype=jnp.float32),
            )
            filt_raw = DrafterSpeculation.filter_logits_for_sampling(logits, **targs)
            filt_lsm = DrafterSpeculation.filter_logits_for_sampling(log_probs, **targs)
            lp_raw = jax.nn.log_softmax(filt_raw, axis=-1)
            lp_lsm = jax.nn.log_softmax(filt_lsm, axis=-1)
            # Probabilities identical (masked positions collapse to 0 either way).
            p_raw = np.asarray(jnp.exp(lp_raw))
            p_lsm = np.asarray(jnp.exp(lp_lsm))
            assert np.allclose(p_raw, p_lsm, atol=1e-6), (
                f"filtered distribution diverged for {params}: max={np.max(np.abs(p_raw - p_lsm))}"
            )
            # Same sampled token under a shared key.
            tok_raw = int(np.asarray(jax.random.categorical(sk, lp_raw)).reshape(-1)[0])
            tok_lsm = int(np.asarray(jax.random.categorical(sk, lp_lsm)).reshape(-1)[0])
            assert tok_raw == tok_lsm, f"sampled token diverged for {params}: {tok_raw} != {tok_lsm}"
    print("  PASS")


if __name__ == "__main__":
    test_verify_sampled_window_reproduces_target_distribution_no_filter()
    test_verify_sampled_window_reproduces_target_distribution_topp()
    test_verify_sampled_window_position0_marginal_k2()
    test_raw_logits_equivalent_to_log_softmax_under_filter()
