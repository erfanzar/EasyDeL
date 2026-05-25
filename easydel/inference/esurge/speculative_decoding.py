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

# ruff: noqa: F821

"""Speculative decoding driver for EasyDeL inference.

Implements the Leviathan-et-al. speculative-decoding loop on top of a
target model + an inline or standalone drafter. Unlike the generic
:class:`easydel.inference.esurge.eSurgeRunner` (which threads
``spec_token_ids`` through the scheduler as opaque slots without any
verification logic), this driver is *dynamic*: every step the drafter
is actually CALLED with the target's last hidden state, draft tokens
are generated, the target verifies them in a single batched forward,
and accepted tokens are emitted.

Two ready-to-use drivers:

- :class:`SpeculativeMTPDriver` — inline MTP heads (Qwen3.5,
  DeepSeek-V3). The drafter shares params with the target; one
  forward pass on the target's last hidden state + the most-recent
  token embedding produces a draft, repeated ``N`` times.
- :class:`SpeculativeAssistantDriver` — standalone drafter models
  (Gemma4 Assistant). The cross-model K/V controller is still a TODO
  (see :meth:`SpeculativeAssistantDriver._draft_tokens`), but the
  draft/verify loop is identical.

Acceptance uses the simple "longest-prefix" check (greedy
verification): a draft at position ``i`` is accepted iff the target's
``argmax`` at position ``i`` equals the draft. Greedy verification is
distribution-correct when the target itself is decoded greedily. For
temperature-sampled targets, set ``greedy=False`` to use Leviathan
rejection sampling via
:func:`easydel.inference.speculative.accept_or_reject`.

CACHE MODEL
-----------
The driver has two execution modes, selected by ``generate(...,
use_cache=...)``:

- **Cache-free** (``use_cache=False``, default): each verify step
  does a full-sequence forward over a fixed-size buffer. Simple and
  provably correct; per-step cost ``O(buffer_length)``.

- **Commit-based KV cache** (``use_cache=True``): IMPLEMENTED BUT
  GATED. The design: the verify forward extends the committed KV
  cache by the ``N`` draft tokens and its returned cache is
  **discarded**; the commit forward re-extends the *same immutable*
  committed cache by only the ``k+1`` accepted tokens. Per-step cost
  ``O(N + k)``.

The commit-based design is the correct way to solve the classic
spec-decode "cache rollback" problem for a hybrid model. Naive
rollback (rewinding per-layer write indices) is *impossible* for
Qwen3.5, where 75% of layers are linear-attention with a **recurrent
state** — a mutated fixed-size summary that cannot be rewound. The
commit-based design sidesteps that: the verify forward's output cache
is thrown away and the commit forward re-derives state from the
immutable pre-step cache, so the recurrent state is *recomputed*
through exactly the accepted tokens. JAX functional purity makes the
"rollback" free.

HOWEVER, the cached path is currently **gated off**
(``use_cache=True`` raises ``NotImplementedError``): a direct
``model(chunk, past_key_values=cache)`` call cannot numerically-
correctly EXTEND an EasyDeL cache. The per-step ``cache_metadata``
(``HybridMetadata`` for Qwen3.5's hybrid stack) that the esurge
executor builds is required; without it the prefill is correct but
cache *extension* desynchronises — proven by the cache-consistency
test. ``_generate_cached`` / ``_cached_forward`` are the ready
structural foundation; completing the path means threading the
esurge-executor cache metadata into ``_cached_forward``. Until then
the **cache-free path is the verified default**.

PERFORMANCE CONTRACT (per step)
-------------------------------
    tokens_emitted_per_step ∈ [1, N+1]   (== 1 + num_accepted)
    target_forwards_per_step = 1 (cache-free) | 2 (cached: verify+commit)

    algorithmic_speedup = mean(tokens_emitted_per_step)
                        = 1 + mean(num_accepted)

For Qwen3.5 single-depth MTP (N=1) at acceptance rate α, the speedup
ceiling is ``1 + α`` ≈ 1.7x at α≈0.7. For Gemma4 Assistant (N=4) the
ceiling is ``1 + 4α`` capped at N+1=5, ≈ 3x at α≈0.7. Random-init
drafters give α≈0 and speedup≈1.0x (no benefit) — that is the
expected result before the drafter is trained.
"""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from easydel.inference.speculative import (
    DrafterProtocol,
    accept_or_reject,
    resample_rejected,
)


@dataclass
class SpecDecodeStats:
    """Per-generation statistics for speculative decoding.

    Attributes:
        num_target_forwards: Number of target-model forward passes
            (1 prefill + 1 per spec-decode step). Without spec-decode
            a baseline does 1 forward per emitted token.
        num_draft_steps: Number of spec-decode steps taken.
        num_drafts_generated: Total draft tokens proposed.
        num_drafts_accepted: Total draft tokens accepted by the
            target's verification.
        wallclock_s: Wallclock seconds for the generation.
        tokens_generated: Total accepted + corrected tokens emitted.
    """

    num_target_forwards: int = 0
    num_draft_steps: int = 0
    num_drafts_generated: int = 0
    num_drafts_accepted: int = 0
    wallclock_s: float = 0.0
    tokens_generated: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed drafts the target accepted."""
        if self.num_drafts_generated == 0:
            return 0.0
        return self.num_drafts_accepted / self.num_drafts_generated

    @property
    def tokens_per_second(self) -> float:
        """End-to-end decoding throughput."""
        if self.wallclock_s == 0:
            return 0.0
        return self.tokens_generated / self.wallclock_s

    @property
    def speedup_vs_baseline(self) -> float:
        """Algorithmic speedup vs a 1-token-per-forward baseline.

        Equal to ``tokens_generated / num_target_forwards``. A
        no-spec-decode baseline emits exactly 1 token per target
        forward, so this ratio is the hardware-independent speedup
        from parallel verification. ``1.0`` means no benefit.
        """
        if self.num_target_forwards == 0:
            return 0.0
        return self.tokens_generated / self.num_target_forwards

    @property
    def mean_accepted_per_step(self) -> float:
        """Average accepted drafts per spec-decode step."""
        if self.num_draft_steps == 0:
            return 0.0
        return self.num_drafts_accepted / self.num_draft_steps




class _SpecDecodeDriverBase:
    """Common machinery for spec-decode drivers (cache-free)."""

    def __init__(
        self,
        target_model: typing.Any,
        drafter: DrafterProtocol,
        num_draft_tokens: int = 4,
        greedy: bool = True,
        rng_seed: int = 0,
    ):
        """Build the driver.

        Args:
            target_model: Full target model. Callable; its output
                must expose ``logits`` and ``last_hidden_state``.
            drafter: ``Qwen3_5MTPDrafter`` or ``Gemma4AssistantDrafter``.
            num_draft_tokens: Speculative tokens proposed per step.
            greedy: Argmax sampling + exact-match verification when
                ``True``; temperature sampling + Leviathan rejection
                sampling when ``False``.
            rng_seed: PRNG seed.
        """
        self.target = target_model
        self.drafter = drafter
        self.num_draft_tokens = int(num_draft_tokens)
        self.greedy = bool(greedy)
        self._rng = jax.random.PRNGKey(rng_seed)
        self._jit_forward: typing.Callable | None = None
        self._jit_cached_forward: dict[int, typing.Callable] = {}

    def _split_rng(self, n: int = 1) -> list[jax.Array]:
        """Advance the driver's PRNG and return ``n`` subkeys.

        Splits the driver's internal PRNG state into ``n + 1`` fresh
        keys, keeps the first key as the next ``self._rng`` and returns
        the remaining ``n`` keys for one-shot use by the caller.

        Args:
            n: Number of subkeys to return.

        Returns:
            List of ``n`` JAX PRNG keys ready for use as sampling RNGs.
        """
        keys = jax.random.split(self._rng, n + 1)
        self._rng = keys[0]
        return list(keys[1:])

    def _target_forward(
        self,
        input_ids: Int[Array, "batch seq"],
    ) -> tuple[Float[Array, "batch seq vocab"], Float[Array, "batch seq hidden"]]:
        """Fixed-shape forward through the target model (JIT-cached).

        The driver always calls this with the same ``(B, max_length)``
        buffer shape, so the jitted function compiles once. Returns
        ``(logits, last_hidden_state)`` — both ``(B, S, *)``.
        """
        if self._jit_forward is None:
            eager = self.target(input_ids=input_ids)
            if getattr(eager, "logits", None) is None or getattr(eager, "last_hidden_state", None) is None:
                raise RuntimeError("target output must expose 'logits' and 'last_hidden_state' for spec-decode")

            @jax.jit
            def _fwd(ids):
                out = self.target(input_ids=ids)
                return out.logits, out.last_hidden_state

            self._jit_forward = _fwd
        return self._jit_forward(input_ids)

    def _cached_forward(
        self,
        input_ids: Int[Array, "batch chunk"],
        cache: typing.Any,
        position_offset: int = 0,
    ) -> tuple[Float[Array, "batch chunk vocab"], Float[Array, "batch chunk hidden"], typing.Any]:
        """Cached forward: extend ``cache`` by an ``input_ids`` chunk.

        Used by the commit-based cache path. The target processes only
        the ``chunk`` new tokens (the cache supplies the prefix's K/V
        and recurrent state), so each call is cheap. JAX functional
        purity means ``cache`` is NOT mutated — the updated cache is
        returned. The driver exploits this for free "rollback": the
        verify forward's returned cache is discarded, and the commit
        forward re-derives state from the same immutable input cache,
        so rejected drafts never pollute the committed cache (this
        also fixes the hybrid-architecture problem — the linear-
        attention recurrent state is re-derived, not rewound).

        Compiles once per distinct chunk length.

        Returns:
            ``(logits, last_hidden_state, updated_cache)``.
        """
        from spectrax import common_types

        batch, chunk_len = input_ids.shape
        position_ids = jnp.arange(chunk_len, dtype=jnp.int32)[None, :].repeat(batch, axis=0) + position_offset
        fn = self._jit_cached_forward.get(chunk_len)
        if fn is None:
            mode = common_types.MODE_PREFILL
            _ = self.target(input_ids=input_ids, past_key_values=cache, position_ids=position_ids, mode=mode)

            @jax.jit
            def _fwd(ids, c, pos):
                out = self.target(input_ids=ids, past_key_values=c, position_ids=pos, mode=mode)
                return out.logits, out.last_hidden_state, out.past_key_values

            fn = _fwd
            self._jit_cached_forward[chunk_len] = fn
        return fn(input_ids, cache, position_ids)

    def _sample(
        self,
        logits: Float[Array, "batch vocab"],
        rng_key: jax.Array,
    ) -> tuple[Int[Array, "batch"], Float[Array, "batch vocab"]]:
        """Sample a token per batch element + return its log-probs."""
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        if self.greedy:
            return jnp.argmax(logits, axis=-1).astype(jnp.int32), log_probs
        return jax.random.categorical(rng_key, logits.astype(jnp.float32)).astype(jnp.int32), log_probs

    def _draft_tokens(
        self,
        seed_token: Int[Array, "batch"],
        seed_hidden: Float[Array, "batch hidden"],
    ) -> tuple[
        Int[Array, "batch n"],
        Float[Array, "batch n"],
        Float[Array, "batch n vocab"] | None,
    ]:
        """Generate ``num_draft_tokens`` drafts per batch element.

        Calls the drafter ``N`` times, chaining each drafted token
        into the next call's input. For inline MTP the same
        ``seed_hidden`` is reused for every depth (single-depth MTP);
        the chained ``input_ids`` carry the new-token signal.

        Args:
            seed_token: The most-recent accepted token ``(B,)``.
            seed_hidden: Target hidden state at the ``seed_token``
                position ``(B, H)``.

        Returns:
            ``(draft_tokens [B, N], draft_log_probs [B, N],
            draft_full_log_probs [B, N, V] | None)``.
        """
        B = seed_token.shape[0]
        cur_token = seed_token  # (B,)
        seed_hidden_bsh = seed_hidden[:, None, :]  # (B, 1, H) — fixed shape
        drafts: list[jax.Array] = []
        lps: list[jax.Array] = []
        fulls: list[jax.Array] = []
        for _ in range(self.num_draft_tokens):
            rng_key = self._split_rng()[0]
            step = self.drafter.draft(
                input_ids=cur_token[:, None],
                target_hidden_states=seed_hidden_bsh,
                sample=not self.greedy,
                rng_key=rng_key,
            )
            drafts.append(step.token_ids)
            lps.append(step.log_probs if step.log_probs is not None else jnp.zeros((B,), dtype=jnp.float32))
            if step.full_log_probs is not None:
                fulls.append(step.full_log_probs)
            cur_token = step.token_ids  # chain the drafted token
        draft_tokens = jnp.stack(drafts, axis=-1)
        draft_lps = jnp.stack(lps, axis=-1)
        draft_fulls = jnp.stack(fulls, axis=-2) if fulls else None
        return draft_tokens, draft_lps, draft_fulls

    def _verify(
        self,
        buf: Int[Array, "batch max_length"],
        cur: int,
        draft_tokens: Int[Array, "batch n"],
        draft_fulls: Float[Array, "batch n vocab"] | None,
    ) -> tuple[Int[Array, "batch num_emitted"], int, Float[Array, "batch hidden"]]:
        """Verify drafts with one fixed-shape target forward.

        ``buf`` is the fixed ``(B, max_length)`` token buffer; ``cur``
        valid tokens occupy ``buf[:, :cur]`` and ``buf[:, cur-1]`` is
        the most-recent token ``T0``. Drafts ``D1..D_{N-1}`` are
        written into ``buf[:, cur:cur+N-1]`` before the forward so the
        target predicts a token at each draft position. Right-padding
        plus causal masking means the trailing pad slots never affect
        the valid positions' logits.

        Args:
            buf: Fixed-size right-padded token buffer.
            cur: Number of valid tokens in ``buf``.
            draft_tokens: Proposed drafts ``(B, N)``.
            draft_fulls: Drafter dense log-probs ``(B, N, V)`` (only
                used for non-greedy rejection sampling).

        Returns:
            ``(emitted_tokens [B, k+1], num_accepted, boundary_hidden)``.
        """
        N = self.num_draft_tokens
        verify_buf = buf.at[:, cur : cur + N].set(draft_tokens)
        logits, hidden = self._target_forward(verify_buf)
        pred0 = cur - 1
        rng_keys = self._split_rng(N + 2)
        target_tok = []
        target_full = []
        for i in range(N + 1):  # N draft checks + 1 bonus
            samp, lp = self._sample(logits[:, pred0 + i, :], rng_keys[i])
            target_tok.append(samp)
            target_full.append(lp)
        target_tok = jnp.stack(target_tok, axis=-1)  # (B, N+1)
        target_full = jnp.stack(target_full, axis=-2)  # (B, N+1, V)

        if self.greedy:
            matches = (target_tok[:, :N] == draft_tokens).astype(jnp.int32)
            accepted_run = jnp.cumprod(matches, axis=-1)
            num_accepted = int(jnp.min(jnp.sum(accepted_run, axis=-1)))
            emitted = jnp.concatenate(
                [draft_tokens[:, :num_accepted], target_tok[:, num_accepted : num_accepted + 1]],
                axis=-1,
            )
        else:
            if draft_fulls is None:
                raise RuntimeError(
                    "Non-greedy spec-decode needs drafter full_log_probs; "
                    "centroid-head drafters are sparse — use greedy=True."
                )
            emitted_cols: list[jax.Array] = []
            num_accepted = 0
            for i in range(N):
                rng_a, rng_b = self._split_rng(2)
                tok = draft_tokens[:, i]
                d_lp = jnp.take_along_axis(draft_fulls[:, i, :], tok[:, None], axis=-1).squeeze(-1)
                t_lp = jnp.take_along_axis(target_full[:, i, :], tok[:, None], axis=-1).squeeze(-1)
                accept_i = accept_or_reject(d_lp, t_lp, rng_a)
                if int(jnp.min(accept_i)) == 1:
                    emitted_cols.append(tok)
                    num_accepted += 1
                else:
                    emitted_cols.append(resample_rejected(target_full[:, i, :], draft_fulls[:, i, :], rng_b))
                    break
            else:
                emitted_cols.append(target_tok[:, N])
            emitted = jnp.stack(emitted_cols, axis=-1)

        boundary_idx = min(pred0 + num_accepted, cur + N - 1)
        boundary_hidden = hidden[:, boundary_idx, :]
        return emitted, num_accepted, boundary_hidden

    def generate(
        self,
        prompt_ids: Int[Array, "batch seq"],
        max_new_tokens: int,
        max_length: int | None = None,
        pad_token_id: int = 0,
        early_stop_token: int | None = None,
        use_cache: bool = False,
    ) -> tuple[Int[Array, "batch out_seq"], SpecDecodeStats]:
        """Run end-to-end speculative-decode generation.

        Two execution modes:

        - ``use_cache=False`` (default): cache-free. Each verify step
          does a full forward over a FIXED-size right-padded buffer
          (compiles once; causal masking ignores the pad slots).
          Simple and provably correct, but per-step cost is
          ``O(buffer_length)``.
        - ``use_cache=True``: commit-based KV cache. The verify
          forward extends the cache by the ``N`` draft tokens (its
          returned cache is DISCARDED); the commit forward re-extends
          the *same immutable* input cache by only the ``k+1``
          accepted tokens. Per-step cost is ``O(N + k)`` — much
          cheaper for long generations. The discard/re-derive pattern
          gives correct rollback for free, including for the hybrid
          linear-attention recurrent state (it is re-derived from the
          committed cache, never rewound). See :meth:`_cached_forward`.

        Args:
            prompt_ids: Prompt tokens ``(B, S)``.
            max_new_tokens: Minimum tokens to generate (the final step
                may overshoot by up to ``N``).
            max_length: Total buffer length (cache-free mode) / cache
                capacity (cached mode). Defaults to
                ``prompt_len + max_new_tokens + N + 4``.
            pad_token_id: Token used for right-padding the buffer.
            early_stop_token: Optional EOS id; stops when emitted.
            use_cache: Select the commit-based KV-cache path.

        Returns:
            ``(output_ids, stats)`` — ``output_ids`` is the valid
            ``(B, prompt + generated)`` slice.
        """
        if use_cache:
            raise NotImplementedError(
                "use_cache=True (commit-based KV cache) is implemented but NOT "
                "yet numerically verified and is gated off. A direct "
                "`model(chunk, past_key_values=cache)` call cannot correctly "
                "EXTEND an EasyDeL cache: the per-step `cache_metadata` "
                "(HybridMetadata for Qwen3.5's hybrid stack) that the esurge "
                "executor builds is required, and without it cache extension "
                "desynchronises (the prefill is correct, extension is not — "
                "proven by the cache-consistency test). Use the verified "
                "cache-free path (use_cache=False). Completing the cached "
                "path means threading esurge-executor cache metadata into "
                "`_cached_forward`; the `_generate_cached` / `_cached_forward` "
                "code below is the ready structural foundation for that work."
            )
        stats = SpecDecodeStats()
        t0 = time.time()
        B, prompt_len = prompt_ids.shape
        N = self.num_draft_tokens
        if max_length is None:
            max_length = prompt_len + max_new_tokens + N + 4
        buf = jnp.full((B, max_length), pad_token_id, dtype=prompt_ids.dtype)
        buf = buf.at[:, :prompt_len].set(prompt_ids)
        cur = prompt_len

        logits, hidden = self._target_forward(buf)
        stats.num_target_forwards += 1
        T0, _ = self._sample(logits[:, cur - 1, :], self._split_rng()[0])
        seed_hidden = hidden[:, cur - 1, :]
        buf = buf.at[:, cur].set(T0)
        cur += 1
        stats.tokens_generated += 1

        if early_stop_token is not None and bool(jnp.any(T0 == early_stop_token)):
            stats.wallclock_s = time.time() - t0
            return buf[:, :cur], stats

        while stats.tokens_generated < max_new_tokens and cur + N + 1 < max_length:
            seed_token = buf[:, cur - 1]
            draft_tokens, _lps, draft_fulls = self._draft_tokens(seed_token, seed_hidden)
            stats.num_draft_steps += 1
            stats.num_drafts_generated += N

            emitted, num_accepted, boundary_hidden = self._verify(buf, cur, draft_tokens, draft_fulls)
            stats.num_target_forwards += 1
            stats.num_drafts_accepted += num_accepted

            emitted_count = int(emitted.shape[-1])
            buf = buf.at[:, cur : cur + emitted_count].set(emitted)
            cur += emitted_count
            stats.tokens_generated += emitted_count
            seed_hidden = boundary_hidden

            if early_stop_token is not None and bool(jnp.any(emitted == early_stop_token)):
                break

        stats.wallclock_s = time.time() - t0
        return buf[:, :cur], stats

    def _generate_cached(
        self,
        prompt_ids: Int[Array, "batch seq"],
        max_new_tokens: int,
        max_length: int | None,
        early_stop_token: int | None,
    ) -> tuple[Int[Array, "batch out_seq"], SpecDecodeStats]:
        """Commit-based KV-cache speculative decode (see :meth:`generate`).

        Per step:

        1. ``verify``: extend the committed cache by the ``N`` drafts.
           The returned cache is DISCARDED.
        2. Decide ``k`` accepted via greedy longest-prefix match.
        3. ``commit``: extend the SAME committed cache by the ``k+1``
           emitted tokens. This becomes the new committed cache.

        Because step 1's cache is discarded and step 3 re-derives from
        the immutable pre-step cache, rejected drafts never corrupt
        the committed state — correct rollback with no index surgery,
        valid even for hybrid linear-attention recurrent state.
        """
        stats = SpecDecodeStats()
        t0 = time.time()
        B, prompt_len = prompt_ids.shape
        N = self.num_draft_tokens
        if max_length is None:
            max_length = prompt_len + max_new_tokens + N + 4

        cache = self.target.init_cache(batch_size=B, max_length=max_length)

        logits, hidden, cache = self._cached_forward(prompt_ids, cache, position_offset=0)
        stats.num_target_forwards += 1
        committed_len = prompt_len  # tokens whose K/V is in the cache
        pending_token = prompt_ids[:, -1]
        pending_logits = logits[:, -1, :]
        pending_hidden = hidden[:, -1, :]
        out_tokens = prompt_ids

        while stats.tokens_generated < max_new_tokens:
            draft_tokens, _lps, _draft_fulls = self._draft_tokens(pending_token, pending_hidden)
            stats.num_draft_steps += 1
            stats.num_drafts_generated += N

            vlogits, _vhidden, _vcache = self._cached_forward(draft_tokens, cache, position_offset=committed_len)
            stats.num_target_forwards += 1
            del _vcache

            keys = self._split_rng(N + 1)
            tgt = [self._sample(pending_logits, keys[0])[0]]
            for i in range(N):
                tgt.append(self._sample(vlogits[:, i, :], keys[i + 1])[0])
            tgt = jnp.stack(tgt, axis=-1)  # (B, N+1)

            matches = (draft_tokens == tgt[:, :N]).astype(jnp.int32)
            accepted_run = jnp.cumprod(matches, axis=-1)
            k = int(jnp.min(jnp.sum(accepted_run, axis=-1)))
            stats.num_drafts_accepted += k
            emitted = jnp.concatenate([draft_tokens[:, :k], tgt[:, k : k + 1]], axis=-1)

            clogits, chidden, cache = self._cached_forward(emitted, cache, position_offset=committed_len)
            stats.num_target_forwards += 1

            emitted_count = int(emitted.shape[-1])
            committed_len += emitted_count
            stats.tokens_generated += emitted_count
            out_tokens = jnp.concatenate([out_tokens, emitted], axis=-1)
            pending_token = emitted[:, -1]
            pending_logits = clogits[:, -1, :]
            pending_hidden = chidden[:, -1, :]

            if early_stop_token is not None and bool(jnp.any(emitted == early_stop_token)):
                break

        stats.wallclock_s = time.time() - t0
        return out_tokens, stats




class SpeculativeMTPDriver(_SpecDecodeDriverBase):
    """Speculative decoding driven by an INLINE MTP head.

    The drafter shares parameters with the target (Qwen3.5 style).
    ``num_draft_tokens=1`` is the published Qwen3.5 setting (single
    MTP depth); ``2-4`` extends drafting by chaining the head, at the
    cost of extra drafter forwards and lower per-token acceptance.
    """

    def __init__(
        self,
        target_model: typing.Any,
        num_draft_tokens: int = 1,
        greedy: bool = True,
        rng_seed: int = 0,
    ):
        """Build the MTP spec-decode driver.

        Args:
            target_model: ``Qwen3_5ForCausalLM`` (or compatible) with
                ``has_mtp() == True``.
            num_draft_tokens: Drafts per step.
            greedy: Argmax verification.
            rng_seed: PRNG seed.

        Raises:
            ValueError: If the target has no MTP head.
        """
        if not getattr(target_model, "has_mtp", lambda: False)():
            raise ValueError("SpeculativeMTPDriver requires a target with config.mtp_num_hidden_layers > 0.")
        super().__init__(
            target_model=target_model,
            drafter=target_model.drafter(method="mtp", num_draft_tokens=num_draft_tokens),
            num_draft_tokens=num_draft_tokens,
            greedy=greedy,
            rng_seed=rng_seed,
        )




def default_assistant_layer_mapping(
    num_assistant_layers: int,
    num_target_layers: int,
) -> list[int]:
    """Heuristic assistant-layer → target-layer mapping.

    The Gemma4 Assistant's Q-only attention reads K/V from the target
    model. Each assistant layer must be told which target layer's K/V
    to attend to. Google's training defines the canonical mapping; it
    is **not** published in the open ``Gemma4Assistant`` config, so
    this heuristic is used until a reference is available:

    Map the ``i``-th assistant layer to the ``i``-th target layer
    counting from the END of the target stack — i.e. the assistant's
    final layer attends to the target's final layer, and earlier
    assistant layers attend to progressively earlier target layers.
    Rationale: the assistant is a shallow drafter whose job is to
    mimic the target's *late* representations, which carry the
    next-token signal.

    Args:
        num_assistant_layers: Drafter layer count (4 for published
            Gemma4 assistants).
        num_target_layers: Target model decoder layer count.

    Returns:
        A list of length ``num_assistant_layers`` mapping each
        assistant layer index to a target layer index.
    """
    if num_target_layers < num_assistant_layers:
        return [num_target_layers - 1] * num_assistant_layers
    base = num_target_layers - num_assistant_layers
    return [base + i for i in range(num_assistant_layers)]


def build_target_kv_pairs(
    target_cache: typing.Any,
    layer_mapping: list[int],
    *,
    page_tables: typing.Sequence[typing.Any] | None = None,
    layer_to_group: dict[int, int] | None = None,
    batch_index: int = 0,
    kv_len: int | None = None,
) -> list[tuple[Array, Array] | None]:
    """Extract per-assistant-layer ``(K, V)`` from a target KV cache.

    The Gemma4 Assistant consumes a ``target_key_value_pairs`` list —
    one ``(K, V)`` tuple per assistant layer — sourced from the target
    model's KV cache. This controller reads the target cache's
    per-layer views and gathers the K/V tensors for the target layers
    named in ``layer_mapping``.

    Args:
        target_cache: The target model's KV cache. Must expose a
            ``views`` sequence whose entries have either ``.key`` /
            ``.value`` attributes (``TransformerCache`` /
            ``HybridCache`` shape) or paged-cache ``.key_pages`` /
            ``.value_pages`` attributes.
        layer_mapping: ``layer_mapping[i]`` is the target layer index
            whose K/V feeds assistant layer ``i``.
        page_tables: Optional eSurge page tables, one per cache group,
            used to gather K/V from ragged paged-cache views.
        layer_to_group: Optional target-layer -> page-table-group map.
            If omitted and paged K/V is encountered, group ``0`` is
            used only when a single page table is supplied.
        batch_index: Request row to gather from non-paged or paged
            caches.
        kv_len: Number of tokens to gather. Defaults to the full
            dense-cache length or one page table row's page coverage.

    Returns:
        A list of ``(K, V)`` tuples (or ``None`` for any target layer
        that has no K/V — e.g. a linear-attention layer in a hybrid
        target). The list length equals ``len(layer_mapping)``.

    Raises:
        ValueError: If ``target_cache`` has no ``views`` attribute.
    """
    views = getattr(target_cache, "views", None)
    if views is None:
        raise ValueError(
            "target_cache must expose a 'views' sequence of per-layer "
            "cache views with .key/.value (got "
            f"{type(target_cache).__name__})"
        )
    pairs: list[tuple[Array, Array] | None] = []
    for tgt_idx in layer_mapping:
        if tgt_idx < 0 or tgt_idx >= len(views):
            pairs.append(None)
            continue
        view = views[tgt_idx]
        view = getattr(view, "transformer", view)
        k = getattr(view, "key", None)
        v = getattr(view, "value", None)
        if k is None or v is None:
            key_pages = getattr(view, "key_pages", None)
            value_pages = getattr(view, "value_pages", None)
            if key_pages is None or value_pages is None or page_tables is None:
                pairs.append(None)
                continue

            group_idx = None
            if layer_to_group is not None:
                group_idx = layer_to_group.get(int(tgt_idx))
            elif len(page_tables) == 1:
                group_idx = 0
            if group_idx is None or group_idx < 0 or group_idx >= len(page_tables):
                pairs.append(None)
                continue

            page_table = page_tables[group_idx]
            page_table_cpu = page_table.get_cpu_tensor() if hasattr(page_table, "get_cpu_tensor") else page_table
            page_size = int(getattr(getattr(view, "metadata", None), "page_size", 1) or 1)
            page_size = max(1, page_size)
            if kv_len is None:
                gather_len = int(page_table_cpu.shape[1]) * page_size
            else:
                gather_len = int(kv_len)
            pages_needed = min(int(page_table_cpu.shape[1]), max(1, (gather_len + page_size - 1) // page_size))
            gather_len = min(gather_len, pages_needed * page_size)
            page_ids = jnp.asarray(page_table_cpu[int(batch_index), :pages_needed], dtype=jnp.int32)
            token_offsets = jnp.arange(gather_len, dtype=jnp.int32)
            page_indices = page_ids[token_offsets // page_size]
            offsets = token_offsets % page_size
            pairs.append((key_pages[page_indices, offsets][None, ...], value_pages[page_indices, offsets][None, ...]))
        else:
            if kv_len is None:
                pairs.append((k[int(batch_index) : int(batch_index) + 1], v[int(batch_index) : int(batch_index) + 1]))
            else:
                pairs.append(
                    (
                        k[int(batch_index) : int(batch_index) + 1, : int(kv_len)],
                        v[int(batch_index) : int(batch_index) + 1, : int(kv_len)],
                    )
                )
    return pairs


class SpeculativeAssistantDriver(_SpecDecodeDriverBase):
    """Speculative decoding driven by a standalone drafter model.

    Built for Gemma4 Assistant. The draft/verify loop is identical to
    :class:`SpeculativeMTPDriver`; the distinguishing piece is the
    cross-model K/V controller — the assistant's Q-only attention
    reads K/V from the target model's KV cache.

    The assistant-layer → target-layer mapping is **configurable**
    via ``layer_mapping``. The published Gemma4 Assistant config does
    not expose Google's canonical mapping, so the default is the
    documented heuristic :func:`default_assistant_layer_mapping`
    (assistant layer ``i`` ← target's ``i``-th-from-last layer).
    Supply ``layer_mapping`` explicitly once the reference is known.
    """

    def __init__(
        self,
        target_model: typing.Any,
        assistant_model: typing.Any,
        target_embed_module: typing.Any | None = None,
        num_draft_tokens: int = 4,
        greedy: bool = True,
        rng_seed: int = 0,
        layer_mapping: list[int] | None = None,
    ):
        """Build the standalone-drafter spec-decode driver.

        Args:
            target_model: The full target model (e.g. Gemma4 31B).
            assistant_model: ``Gemma4AssistantForCausalLM`` drafter.
            target_embed_module: ``embed_tokens`` of the target;
                defaults to ``target_model.get_input_embeddings()``.
            num_draft_tokens: Drafts per step.
            greedy: Argmax verification.
            rng_seed: PRNG seed.
            layer_mapping: Optional explicit assistant-layer →
                target-layer mapping. ``None`` uses
                :func:`default_assistant_layer_mapping`.
        """
        super().__init__(
            target_model=target_model,
            drafter=target_model.drafter(
                method="gemma4_assistant",
                assistant_model=assistant_model,
                target_embed_module=target_embed_module,
                num_draft_tokens=num_draft_tokens,
                layer_mapping=layer_mapping,
            ),
            num_draft_tokens=num_draft_tokens,
            greedy=greedy,
            rng_seed=rng_seed,
        )
        self.assistant_model = assistant_model
        self._explicit_layer_mapping = layer_mapping
        self._target_cache: typing.Any | None = None

    def resolve_layer_mapping(self) -> list[int]:
        """Resolve the assistant→target layer mapping in use.

        Returns:
            An explicit mapping if one was supplied to ``__init__``,
            otherwise the heuristic from
            :func:`default_assistant_layer_mapping`.
        """
        if self._explicit_layer_mapping is not None:
            return self._explicit_layer_mapping
        num_assistant = int(self.assistant_model.config.text_config.num_hidden_layers)
        target_cfg = self.target.config.get_text_config()
        num_target = int(getattr(target_cfg, "num_hidden_layers", num_assistant))
        return default_assistant_layer_mapping(num_assistant, num_target)

    def set_target_cache(self, target_cache: typing.Any) -> None:
        """Register the target's KV cache for the next draft step.

        The driver (or an external runner) calls this after each
        target forward so :meth:`_draft_tokens` can pull cross-model
        K/V. ``None`` resets to the self-K/V fallback.

        Args:
            target_cache: A ``TransformerCache`` / ``HybridCache`` from
                the target model's most recent forward.
        """
        self._target_cache = target_cache

    def _draft_tokens(self, seed_token, seed_hidden):
        """Generate drafts, feeding the target's K/V to the assistant.

        When a target cache has been registered via
        :meth:`set_target_cache`, the cross-model K/V controller
        (:func:`build_target_kv_pairs`) extracts per-assistant-layer
        K/V and the assistant's Q-only attention attends to it. If no
        target cache is available the assistant falls back to self-K/V
        (shape-correct but not target-conditioned).
        """
        if self._target_cache is None:
            return super()._draft_tokens(seed_token, seed_hidden)

        kv_pairs = build_target_kv_pairs(self._target_cache, self.resolve_layer_mapping())
        B = seed_token.shape[0]
        cur_token = seed_token
        seed_hidden_bsh = seed_hidden[:, None, :]
        drafts: list[jax.Array] = []
        lps: list[jax.Array] = []
        fulls: list[jax.Array] = []
        for _ in range(self.num_draft_tokens):
            rng_key = self._split_rng()[0]
            step = self.drafter.draft(
                input_ids=cur_token[:, None],
                target_hidden_states=seed_hidden_bsh,
                target_kv_cache=kv_pairs,
                sample=not self.greedy,
                rng_key=rng_key,
            )
            drafts.append(step.token_ids)
            lps.append(step.log_probs if step.log_probs is not None else jnp.zeros((B,), dtype=jnp.float32))
            if step.full_log_probs is not None:
                fulls.append(step.full_log_probs)
            cur_token = step.token_ids
            if step.hidden_states is not None:
                seed_hidden_bsh = step.hidden_states[:, -1:, :]
        draft_tokens = jnp.stack(drafts, axis=-1)
        draft_lps = jnp.stack(lps, axis=-1)
        draft_fulls = jnp.stack(fulls, axis=-2) if fulls else None
        return draft_tokens, draft_lps, draft_fulls


__all__ = [
    "SpecDecodeStats",
    "SpeculativeAssistantDriver",
    "SpeculativeMTPDriver",
    "build_target_kv_pairs",
    "default_assistant_layer_mapping",
]
