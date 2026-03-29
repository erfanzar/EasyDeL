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
    num_reqs = max(1, int(num_reqs))
    res = int(min_input_pad) if num_reqs <= int(min_input_pad) else 1 << (num_reqs - 1).bit_length()
    return min(int(upper_limit), res)


def main() -> None:
    args = _parse_args()
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32

    @jax.jit
    def legacy_full_sampler(logits, token_history, seq_lens, active_mask, presence, frequency, repetition, rng):
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
    def optimized_full_sampler(logits, token_counts_full, row_indices, active_mask, presence, frequency, repetition, rng):
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
                return sampled, active_mask, updated_counts

            def _scatter_output(_):
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
        "active1_x": results["legacy_full_sampler_active1"]["mean_ms"] / results["optimized_full_sampler_active1"]["mean_ms"],
        "active128_x": (
            results["legacy_full_sampler_active128"]["mean_ms"]
            / results["optimized_full_sampler_active128"]["mean_ms"]
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
