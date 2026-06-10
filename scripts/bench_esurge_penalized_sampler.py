#!/usr/bin/env python3
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

"""Microbenchmark the eSurge penalized sampler legacy/optimized/compacted paths.

Compares three sampling kernels on synthetic logits + token histories:

* ``legacy_full_sampler`` — recomputes per-vocab token counts from the raw
  ``token_history`` array every call (the original eSurge path).
* ``optimized_full_sampler`` — reuses a precomputed ``token_counts`` matrix
  (``[padded_reqs, vocab_size]``) and incrementally updates it after sampling.
* ``compacted_full_sampler`` — like the optimized path but additionally
  gather-compacts ``padded_reqs`` down to the active sub-batch before running
  ``apply_history_penalties_from_counts`` / ``sample_tokens``, then scatters
  results back out. Used by the runtime when most slots in the request pad are
  inactive.

All three are wrapped in ``jax.jit`` and warmed up before timing. Each path is
exercised at two activity levels (1 active req out of ``padded_reqs``, and all
``padded_reqs`` active) so the report shows both the compaction win and the
steady-state cost. Results are printed as JSON to stdout including per-config
mean/min/max latencies and a few speedup ratios.

Usage:
    python scripts/bench_esurge_penalized_sampler.py \\
        --vocab-size 131072 --history-len 8192 --padded-reqs 128 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
from jax import lax

from easydel.inference.esurge.core.sampler import (
    apply_history_penalties,
    apply_history_penalties_from_counts,
    build_history_token_counts,
    sample_tokens,
    update_token_counts,
)
from easydel.inference.esurge.core.sampling_metadata import SamplingMetadata


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sampler benchmark.

    Returns:
        argparse.Namespace: Parsed flags (``vocab_size``, ``history_len``,
            ``padded_reqs``, ``repeats``, ``presence_penalty``,
            ``frequency_penalty``, ``repetition_penalty``, ``temperature``,
            ``top_p``, ``top_k``, ``min_p``, ``sampler_min_pad``, ``dtype``).
    """
    parser = argparse.ArgumentParser(description="Benchmark legacy and optimized eSurge penalized sampler paths.")
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--history-len", type=int, default=8192)
    parser.add_argument("--padded-reqs", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--sampler-min-pad", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    return parser.parse_args()


def _measure(fn, *args, repeats: int) -> dict[str, float | list[float]]:
    """Time a callable ``repeats`` times and return latency statistics.

    Blocks on the function output between iterations via ``jax.block_until_ready``
    so the measured time reflects device-side execution, not just dispatch.

    Args:
        fn: Callable to benchmark. Typically a ``jax.jit``-compiled function.
        *args: Positional arguments forwarded to ``fn`` on every invocation.
        repeats: Number of timed iterations to collect.

    Returns:
        dict[str, float | list[float]]: Mapping with ``mean_ms``, ``min_ms``,
            ``max_ms``, and the raw ``samples_ms`` list.
    """
    samples_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": statistics.mean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def _pad_reqs(num_reqs: int, upper_limit: int, min_input_pad: int) -> int:
    """Round ``num_reqs`` up to a power-of-two pad bucket, clamped to ``upper_limit``.

    Mirrors the bucket-padding policy used by the eSurge runner so the
    benchmark can construct compacted shapes that line up with what would
    actually be JIT-compiled at serving time.

    Args:
        num_reqs: Live request count to pad.
        upper_limit: Maximum padded bucket size (typically ``padded_reqs``).
        min_input_pad: Minimum bucket size; ``num_reqs <= min_input_pad`` snaps
            straight to ``min_input_pad``.

    Returns:
        int: Padded bucket size in ``[min_input_pad, upper_limit]``.
    """
    num_reqs = max(1, int(num_reqs))
    res = int(min_input_pad) if num_reqs <= int(min_input_pad) else 1 << (num_reqs - 1).bit_length()
    return min(int(upper_limit), res)


def main() -> None:
    """Run the eSurge penalized sampler benchmark and print results as JSON.

    Builds synthetic logits, token-history matrices, and penalty vectors at the
    configured ``vocab_size`` / ``history_len`` / ``padded_reqs``, then times
    the legacy, optimized, and compacted sampler kernels for both the
    one-active-slot and all-slots-active cases. The final JSON document
    written to stdout contains the parsed config, per-case latency stats, and
    a ``speedups`` block comparing the optimized/compacted paths against the
    legacy baseline.

    Args:
        None. Reads CLI arguments via :func:`_parse_args`.

    Returns:
        None. The result document is printed to stdout.
    """
    args = _parse_args()
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32

    @jax.jit
    def legacy_full_sampler(logits, token_history, seq_lens, active_mask, presence, frequency, repetition, rng):
        """Legacy sampler: rebuild token counts from ``token_history`` per call.

        Args:
            logits: ``[padded_reqs, vocab_size]`` raw logits matrix.
            token_history: ``[padded_reqs, history_len]`` token IDs seen so far.
            seq_lens: Active history length per request (``[padded_reqs]``).
            active_mask: Boolean mask of active request slots.
            presence: Presence penalty per request.
            frequency: Frequency penalty per request.
            repetition: Repetition penalty per request.
            rng: PRNG key for the sampling step.

        Returns:
            Sampled token IDs as a ``[padded_reqs]`` int array.
        """
        adjusted = apply_history_penalties(
            logits,
            token_history=token_history,
            seq_lens=seq_lens,
            active_mask=active_mask,
            presence_penalties=presence,
            frequency_penalties=frequency,
            repetition_penalties=repetition,
        )
        metadata = SamplingMetadata(
            temperatures=jnp.full((args.padded_reqs, 1), args.temperature, dtype=dtype),
            top_ps=jnp.full((args.padded_reqs,), args.top_p, dtype=dtype),
            top_ks=jnp.full((args.padded_reqs,), args.top_k, dtype=jnp.int32),
            min_ps=jnp.full((args.padded_reqs,), args.min_p, dtype=dtype),
            sampling_seeds=None,
            is_all_greedy=False,
            need_min_p_sampling=args.min_p > 0.0,
            do_penalties=False,
            linear_penalty=None,
        )
        return sample_tokens(adjusted, metadata, rng)

    @jax.jit
    def optimized_full_sampler(
        logits, token_counts_full, row_indices, active_mask, presence, frequency, repetition, rng
    ):
        """Optimized sampler: gather precomputed token counts and update in place.

        Args:
            logits: ``[padded_reqs, vocab_size]`` raw logits matrix.
            token_counts_full: ``[padded_reqs, vocab_size]`` running token
                frequency counters maintained across decode steps.
            row_indices: Indices into ``token_counts_full`` selecting the row
                that backs each request slot.
            active_mask: Boolean mask of active request slots.
            presence: Presence penalty per request.
            frequency: Frequency penalty per request.
            repetition: Repetition penalty per request.
            rng: PRNG key for the sampling step.

        Returns:
            Tuple ``(sampled, updated_counts)`` where ``sampled`` is the
            ``[padded_reqs]`` int array of sampled tokens and
            ``updated_counts`` is the post-update token-count matrix.
        """
        adjusted = apply_history_penalties_from_counts(
            logits,
            token_counts=token_counts_full[row_indices],
            active_mask=active_mask,
            presence_penalties=presence,
            frequency_penalties=frequency,
            repetition_penalties=repetition,
        )
        metadata = SamplingMetadata(
            temperatures=jnp.full((args.padded_reqs, 1), args.temperature, dtype=dtype),
            top_ps=jnp.full((args.padded_reqs,), args.top_p, dtype=dtype),
            top_ks=jnp.full((args.padded_reqs,), args.top_k, dtype=jnp.int32),
            min_ps=jnp.full((args.padded_reqs,), args.min_p, dtype=dtype),
            sampling_seeds=None,
            is_all_greedy=False,
            need_min_p_sampling=args.min_p > 0.0,
            do_penalties=False,
            linear_penalty=None,
        )
        sampled = sample_tokens(adjusted, metadata, rng)
        updated_counts = update_token_counts(
            token_counts_full,
            row_indices=row_indices,
            sampled_tokens=sampled,
            valid_mask=active_mask,
        )
        return sampled, updated_counts

    @jax.jit
    def compacted_full_sampler(
        logits,
        token_counts_full,
        gather_positions,
        sampling_seeds,
        scatter_positions,
        active_mask,
        presence,
        frequency,
        repetition,
        rng,
    ):
        """Compacted sampler: gather active slots, sample, then scatter back.

        Mirrors the runtime fast path that pays attention only to the active
        sub-batch and falls back to a full identity gather when the layouts
        happen to match.

        Args:
            logits: ``[padded_reqs, vocab_size]`` raw logits matrix.
            token_counts_full: ``[padded_reqs, vocab_size]`` running token
                frequency counters.
            gather_positions: Indices into the padded batch selecting the
                active slots (``[sampler_padded_reqs]``).
            sampling_seeds: Per-active-slot PRNG seeds used to derive sampler
                keys.
            scatter_positions: Output positions for writing sampled tokens
                back into a ``padded_reqs + spill`` workspace.
            active_mask: Active mask over the compacted layout.
            presence: Presence penalty per active slot.
            frequency: Frequency penalty per active slot.
            repetition: Repetition penalty per active slot.
            rng: PRNG key for the sampling step.

        Returns:
            Tuple ``(tokens, valid, updated_counts)`` where ``tokens`` is the
            ``[padded_reqs]`` int array of scattered sampled tokens (``-1``
            for inactive slots), ``valid`` is the matching boolean mask, and
            ``updated_counts`` is the refreshed token-count matrix.
        """
        sampler_padded_reqs = gather_positions.shape[0]
        if sampler_padded_reqs == args.padded_reqs:
            identity_layout = jnp.all(gather_positions == jnp.arange(sampler_padded_reqs, dtype=jnp.int32)) & jnp.all(
                scatter_positions == jnp.arange(sampler_padded_reqs, dtype=jnp.int32)
            )
            compact_logits = lax.cond(
                identity_layout,
                lambda _: logits[:sampler_padded_reqs],
                lambda _: logits[gather_positions],
                operand=None,
            )
        else:
            identity_layout = None
            compact_logits = logits[gather_positions]
        compact_counts = token_counts_full[gather_positions]
        adjusted = apply_history_penalties_from_counts(
            compact_logits,
            token_counts=compact_counts,
            active_mask=active_mask,
            presence_penalties=presence,
            frequency_penalties=frequency,
            repetition_penalties=repetition,
        )
        metadata = SamplingMetadata(
            temperatures=jnp.where(
                active_mask[:, None],
                jnp.full((sampler_padded_reqs, 1), args.temperature, dtype=jnp.float32),
                jnp.ones((sampler_padded_reqs, 1), dtype=jnp.float32),
            ),
            top_ps=jnp.where(
                active_mask,
                jnp.full((sampler_padded_reqs,), args.top_p, dtype=jnp.float32),
                jnp.ones((sampler_padded_reqs,), dtype=jnp.float32),
            ),
            top_ks=jnp.where(
                active_mask,
                jnp.full((sampler_padded_reqs,), args.top_k, dtype=jnp.int32),
                jnp.zeros((sampler_padded_reqs,), dtype=jnp.int32),
            ),
            min_ps=jnp.where(
                active_mask,
                jnp.full((sampler_padded_reqs,), args.min_p, dtype=jnp.float32),
                jnp.zeros((sampler_padded_reqs,), dtype=jnp.float32),
            ),
            sampling_seeds=sampling_seeds,
            is_all_greedy=args.temperature <= 0.0,
            need_min_p_sampling=args.min_p > 0.0,
            do_penalties=False,
            linear_penalty=None,
        )
        sampled = sample_tokens(adjusted, metadata, rng)
        updated_counts = update_token_counts(
            token_counts_full,
            row_indices=gather_positions,
            sampled_tokens=sampled,
            valid_mask=active_mask,
        )
        spill = scatter_positions.shape[0]

        if identity_layout is not None:

            def _identity_output(_):
                """Return sampled tokens directly when gather/scatter are identity."""
                return sampled, active_mask, updated_counts

            def _scatter_output(_):
                """Scatter sampled tokens back into the full padded layout."""
                full_tokens = jnp.full((args.padded_reqs + spill,), -1, dtype=jnp.int32)
                full_valid = jnp.zeros((args.padded_reqs + spill,), dtype=jnp.bool_)
                full_tokens_local = full_tokens.at[scatter_positions].set(jnp.where(active_mask, sampled, -1))
                full_valid_local = full_valid.at[scatter_positions].set(active_mask)
                return full_tokens_local[: args.padded_reqs], full_valid_local[: args.padded_reqs], updated_counts

            return lax.cond(identity_layout, _identity_output, _scatter_output, operand=None)

        full_tokens = jnp.full((args.padded_reqs + spill,), -1, dtype=jnp.int32)
        full_valid = jnp.zeros((args.padded_reqs + spill,), dtype=jnp.bool_)
        full_tokens = full_tokens.at[scatter_positions].set(jnp.where(active_mask, sampled, -1))
        full_valid = full_valid.at[scatter_positions].set(active_mask)
        return full_tokens[: args.padded_reqs], full_valid[: args.padded_reqs], updated_counts

    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (args.padded_reqs, args.vocab_size), dtype=dtype)
    token_history = (jnp.arange(args.history_len, dtype=jnp.int32)[None, :] * 17) % args.vocab_size
    token_history = jnp.broadcast_to(token_history, (args.padded_reqs, args.history_len))
    token_counts = build_history_token_counts(
        token_history=token_history,
        seq_lens=jnp.full((args.padded_reqs,), args.history_len, dtype=jnp.int32),
        active_mask=jnp.ones((args.padded_reqs,), dtype=jnp.bool_),
        vocab_size=args.vocab_size,
    )
    row_indices = jnp.arange(args.padded_reqs, dtype=jnp.int32)
    presence = jnp.full((args.padded_reqs,), args.presence_penalty, dtype=dtype)
    frequency = jnp.full((args.padded_reqs,), args.frequency_penalty, dtype=dtype)
    repetition = jnp.full((args.padded_reqs,), args.repetition_penalty, dtype=dtype)

    seq_lens_active_128 = jnp.full((args.padded_reqs,), args.history_len, dtype=jnp.int32)
    active_mask_128 = jnp.ones((args.padded_reqs,), dtype=jnp.bool_)
    seq_lens_active_1 = seq_lens_active_128.at[1:].set(0)
    active_mask_1 = active_mask_128.at[1:].set(False)
    compact_active1_padded = _pad_reqs(1, args.padded_reqs, args.sampler_min_pad)
    compact_active128_padded = _pad_reqs(args.padded_reqs, args.padded_reqs, args.sampler_min_pad)
    compact_gather_active1 = jnp.zeros((compact_active1_padded,), dtype=jnp.int32)
    compact_seeds_active1 = (args.padded_reqs + jnp.arange(compact_active1_padded, dtype=jnp.int32)).at[0].set(0)
    compact_scatter_active1 = compact_seeds_active1
    compact_active_mask_1 = jnp.array([True] + [False] * max(0, compact_active1_padded - 1), dtype=jnp.bool_)
    compact_presence_active1 = jnp.array(
        [args.presence_penalty] + [0.0] * max(0, compact_active1_padded - 1),
        dtype=dtype,
    )
    compact_frequency_active1 = jnp.array(
        [args.frequency_penalty] + [0.0] * max(0, compact_active1_padded - 1),
        dtype=dtype,
    )
    compact_repetition_active1 = jnp.array(
        [args.repetition_penalty] + [1.0] * max(0, compact_active1_padded - 1),
        dtype=dtype,
    )
    compact_gather_active128 = jnp.arange(compact_active128_padded, dtype=jnp.int32)
    compact_seeds_active128 = compact_gather_active128
    compact_scatter_active128 = compact_gather_active128
    compact_active_mask_128 = jnp.ones((compact_active128_padded,), dtype=jnp.bool_)
    compact_presence_active128 = jnp.full((compact_active128_padded,), args.presence_penalty, dtype=dtype)
    compact_frequency_active128 = jnp.full((compact_active128_padded,), args.frequency_penalty, dtype=dtype)
    compact_repetition_active128 = jnp.full((compact_active128_padded,), args.repetition_penalty, dtype=dtype)

    legacy_full_sampler(
        logits, token_history, seq_lens_active_128, active_mask_128, presence, frequency, repetition, key
    ).block_until_ready()
    legacy_full_sampler(
        logits, token_history, seq_lens_active_1, active_mask_1, presence, frequency, repetition, key
    ).block_until_ready()
    jax.block_until_ready(
        optimized_full_sampler(logits, token_counts, row_indices, active_mask_128, presence, frequency, repetition, key)
    )
    jax.block_until_ready(
        optimized_full_sampler(logits, token_counts, row_indices, active_mask_1, presence, frequency, repetition, key)
    )
    jax.block_until_ready(
        compacted_full_sampler(
            logits,
            token_counts,
            compact_gather_active1,
            compact_seeds_active1,
            compact_scatter_active1,
            compact_active_mask_1,
            compact_presence_active1,
            compact_frequency_active1,
            compact_repetition_active1,
            key,
        )
    )
    jax.block_until_ready(
        compacted_full_sampler(
            logits,
            token_counts,
            compact_gather_active128,
            compact_seeds_active128,
            compact_scatter_active128,
            compact_active_mask_128,
            compact_presence_active128,
            compact_frequency_active128,
            compact_repetition_active128,
            key,
        )
    )
    jax.block_until_ready(
        build_history_token_counts(
            token_history=token_history,
            seq_lens=seq_lens_active_128,
            active_mask=active_mask_128,
            vocab_size=args.vocab_size,
        )
    )
    jax.block_until_ready(
        build_history_token_counts(
            token_history=token_history,
            seq_lens=seq_lens_active_1,
            active_mask=active_mask_1,
            vocab_size=args.vocab_size,
        )
    )

    results = {
        "config": {
            "dtype": args.dtype,
            "vocab_size": args.vocab_size,
            "history_len": args.history_len,
            "padded_reqs": args.padded_reqs,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
            "repetition_penalty": args.repetition_penalty,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
        },
        "legacy_build_counts_active1": _measure(
            lambda: build_history_token_counts(
                token_history=token_history,
                seq_lens=seq_lens_active_1,
                active_mask=active_mask_1,
                vocab_size=args.vocab_size,
            ),
            repeats=args.repeats,
        ),
        "legacy_build_counts_active128": _measure(
            lambda: build_history_token_counts(
                token_history=token_history,
                seq_lens=seq_lens_active_128,
                active_mask=active_mask_128,
                vocab_size=args.vocab_size,
            ),
            repeats=args.repeats,
        ),
        "legacy_full_sampler_active1": _measure(
            legacy_full_sampler,
            logits,
            token_history,
            seq_lens_active_1,
            active_mask_1,
            presence,
            frequency,
            repetition,
            key,
            repeats=args.repeats,
        ),
        "legacy_full_sampler_active128": _measure(
            legacy_full_sampler,
            logits,
            token_history,
            seq_lens_active_128,
            active_mask_128,
            presence,
            frequency,
            repetition,
            key,
            repeats=args.repeats,
        ),
        "optimized_full_sampler_active1": _measure(
            optimized_full_sampler,
            logits,
            token_counts,
            row_indices,
            active_mask_1,
            presence,
            frequency,
            repetition,
            key,
            repeats=args.repeats,
        ),
        "optimized_full_sampler_active128": _measure(
            optimized_full_sampler,
            logits,
            token_counts,
            row_indices,
            active_mask_128,
            presence,
            frequency,
            repetition,
            key,
            repeats=args.repeats,
        ),
        "compacted_full_sampler_active1": _measure(
            compacted_full_sampler,
            logits,
            token_counts,
            compact_gather_active1,
            compact_seeds_active1,
            compact_scatter_active1,
            compact_active_mask_1,
            compact_presence_active1,
            compact_frequency_active1,
            compact_repetition_active1,
            key,
            repeats=args.repeats,
        ),
        "compacted_full_sampler_active128": _measure(
            compacted_full_sampler,
            logits,
            token_counts,
            compact_gather_active128,
            compact_seeds_active128,
            compact_scatter_active128,
            compact_active_mask_128,
            compact_presence_active128,
            compact_frequency_active128,
            compact_repetition_active128,
            key,
            repeats=args.repeats,
        ),
    }

    results["speedups"] = {
        "active1_x": results["legacy_full_sampler_active1"]["mean_ms"]
        / results["optimized_full_sampler_active1"]["mean_ms"],
        "active128_x": (
            results["legacy_full_sampler_active128"]["mean_ms"] / results["optimized_full_sampler_active128"]["mean_ms"]
        ),
        "compacted_active1_vs_optimized_x": (
            results["optimized_full_sampler_active1"]["mean_ms"] / results["compacted_full_sampler_active1"]["mean_ms"]
        ),
        "compacted_active1_vs_legacy_x": (
            results["legacy_full_sampler_active1"]["mean_ms"] / results["compacted_full_sampler_active1"]["mean_ms"]
        ),
        "compacted_active128_vs_optimized_x": (
            results["optimized_full_sampler_active128"]["mean_ms"]
            / results["compacted_full_sampler_active128"]["mean_ms"]
        ),
    }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
