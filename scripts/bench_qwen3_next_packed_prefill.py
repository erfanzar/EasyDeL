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

"""Benchmark Qwen3Next packed prefill legacy vs refactored helpers."""

from __future__ import annotations

import argparse
import time
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np

from easydel.modules.qwen3_next.modeling_qwen3_next import (
    _apply_qwen3_next_packed_updates,
    _apply_qwen3_next_packed_updates_legacy,
)
from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig
from easydel.operations import OperationMetadata
from easydel.operations.kernels import GatedDeltaRuleOp

LAYOUT_AXIS_DIMS = {
    "fsdp4": (1, 4, 1, 1, 1),
    "tp4": (1, 1, 1, 4, 1),
}


def _grouped_gdr_decode_jax_only(self, query, key, value, beta, decay, recurrent_state):
    return GatedDeltaRuleOp.grouped_gdr_decode_jax(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
    )


def _grouped_gdr_decode_pallas_only(self, query, key, value, beta, decay, recurrent_state):
    return self.grouped_gdr_decode_shard_map_pallas(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
    )


def _make_gdr_op(layout: str, runtime_dtype=jnp.bfloat16, grouped_decode_backend: str = "auto"):
    axis_dims = LAYOUT_AXIS_DIMS[layout]
    base_config = Qwen3NextConfig(
        sharding_axis_dims=axis_dims,
        backend=jax.default_backend(),
    )
    gdr_op = GatedDeltaRuleOp(
        OperationMetadata(
            runtime_dtype=runtime_dtype,
            runtime_softmax_dtype=jnp.float32,
            platform=jax.default_backend(),
            backend=jax.default_backend(),
            base_config=base_config,
        )
    )
    if grouped_decode_backend == "jax":
        gdr_op.grouped_gdr_decode = MethodType(_grouped_gdr_decode_jax_only, gdr_op)
    elif grouped_decode_backend == "pallas":
        gdr_op.grouped_gdr_decode = MethodType(_grouped_gdr_decode_pallas_only, gdr_op)
    return gdr_op


def _build_schedule(case: str, bucket: int, num_slots: int) -> tuple[np.ndarray, int]:
    if case == "decode_like":
        num_requests = min(num_slots, 32)
        lengths = np.ones((num_requests,), dtype=np.int32)
    elif case == "mixed":
        num_requests = min(num_slots, 32)
        num_single = max(1, num_requests // 4)
        num_multi = max(1, num_requests - num_single)
        remaining = max(bucket - num_single, num_multi)
        multi_len = max(2, remaining // num_multi)
        lengths = np.concatenate(
            [
                np.ones((num_single,), dtype=np.int32),
                np.full((num_multi,), multi_len, dtype=np.int32),
            ]
        )
    elif case == "prefill_heavy":
        num_requests = min(num_slots, 32)
        tokens_per_request = max(2, bucket // max(num_requests, 1))
        lengths = np.full((num_requests,), tokens_per_request, dtype=np.int32)
    else:
        raise ValueError(f"Unknown benchmark case: {case}")

    query_start_loc = np.zeros((num_slots + 1,), dtype=np.int32)
    if num_requests > 0:
        query_start_loc[1 : num_requests + 1] = np.cumsum(lengths, dtype=np.int32)
        query_start_loc[num_requests + 1 :] = query_start_loc[num_requests]
    return query_start_loc, num_requests


def _make_inputs(case: str, bucket: int, num_slots: int, dtype: jnp.dtype) -> dict[str, object]:
    num_k_heads = 4
    head_k_dim = 128
    num_v_heads = 16
    head_v_dim = 128
    key_dim = num_k_heads * head_k_dim
    conv_dim = key_dim * 2 + num_v_heads * head_v_dim
    d_conv = 4

    query_start_loc, num_requests = _build_schedule(case, bucket, num_slots)
    rng = jax.random.key(bucket + 1000 * (1 + ["decode_like", "mixed", "prefill_heavy"].index(case)))

    conv_states = jax.random.normal(rng, (num_slots, conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    recurrent_states = jax.random.normal(
        jax.random.fold_in(rng, 1),
        (num_slots, num_v_heads, head_k_dim, head_v_dim),
        dtype=jnp.float32,
    ).astype(dtype)
    conv_input = jax.random.normal(
        jax.random.fold_in(rng, 2),
        (1, bucket, conv_dim),
        dtype=jnp.float32,
    ).astype(dtype)
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (1, bucket, num_v_heads), dtype=jnp.float32)
    ).astype(dtype)
    decay = (
        -jax.nn.softplus(jax.random.normal(jax.random.fold_in(rng, 4), (1, bucket, num_v_heads), dtype=jnp.float32))
    ).astype(dtype)
    kernel = jax.random.normal(jax.random.fold_in(rng, 5), (conv_dim, d_conv), dtype=jnp.float32).astype(dtype)

    return {
        "conv_states": conv_states,
        "recurrent_states": recurrent_states,
        "conv_input": conv_input,
        "beta": beta,
        "decay": decay,
        "kernel": kernel,
        "query_start_loc": jnp.asarray(query_start_loc, dtype=jnp.int32),
        "num_requests": jnp.asarray(num_requests, dtype=jnp.int32),
        "key_dim": key_dim,
        "num_k_heads": num_k_heads,
        "head_k_dim": head_k_dim,
        "num_v_heads": num_v_heads,
        "head_v_dim": head_v_dim,
        "expand_ratio": num_v_heads // num_k_heads,
        "conv_output_dtype": dtype,
    }


def _block_tree(tree):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def _time_callable(fn, *, warmup: int, repeats: int) -> tuple[object, float]:
    out = None
    for _ in range(warmup):
        out = fn()
        _block_tree(out)

    start = time.perf_counter()
    for _ in range(repeats):
        out = fn()
        _block_tree(out)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return out, elapsed_ms


def _allclose_tree(lhs, rhs, *, rtol: float = 0.02, atol: float = 0.05) -> bool:
    leaves = zip(jax.tree_util.tree_leaves(lhs), jax.tree_util.tree_leaves(rhs), strict=True)
    return all(jnp.allclose(a.astype(jnp.float32), b.astype(jnp.float32), rtol=rtol, atol=atol) for a, b in leaves)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per case.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations per case.")
    parser.add_argument("--num-slots", type=int, default=32, help="Packed slot count.")
    parser.add_argument(
        "--layout",
        choices=tuple(LAYOUT_AXIS_DIMS),
        default="tp4",
        help="Logical mesh layout to benchmark.",
    )
    parser.add_argument(
        "--gdr-backend",
        choices=("auto", "jax", "pallas"),
        default="auto",
        help="Grouped decode backend used by the single-token fast lane.",
    )
    args = parser.parse_args()

    cases = ("decode_like", "mixed", "prefill_heavy")
    buckets = (512, 2048)
    dtype = jnp.bfloat16
    gdr_op = _make_gdr_op(args.layout, runtime_dtype=dtype, grouped_decode_backend=args.gdr_backend)
    mesh = gdr_op.metadata.mesh

    print(
        f"backend={jax.default_backend()} devices={len(jax.devices())} "
        f"layout={args.layout} gdr_backend={args.gdr_backend} mesh_shape={mesh.shape}"
    )
    print(f"warmup={args.warmup} repeats={args.repeats} num_slots={args.num_slots} dtype={dtype}")
    print()
    print("case           bucket   legacy_ms  unified_ms   speedup   outputs_match")

    with mesh:
        for case in cases:
            for bucket in buckets:
                inputs = _make_inputs(case, bucket, args.num_slots, dtype)

                legacy_fn = jax.jit(
                    lambda: _apply_qwen3_next_packed_updates_legacy(
                        **inputs,  # noqa
                        gdr_op=gdr_op,
                    )
                )
                ref_fn = jax.jit(
                    lambda: _apply_qwen3_next_packed_updates(
                        **inputs,  # noqa
                        gdr_op=gdr_op,
                    )
                )

                legacy_out, legacy_ms = _time_callable(legacy_fn, warmup=args.warmup, repeats=args.repeats)
                unified_out, unified_ms = _time_callable(ref_fn, warmup=args.warmup, repeats=args.repeats)
                speedup = ((legacy_ms - unified_ms) / legacy_ms * 100.0) if legacy_ms else 0.0
                outputs_match = _allclose_tree(legacy_out, unified_out)

                print(
                    f"{case:<14} {bucket:>6d} {legacy_ms:>10.2f} {unified_ms:>11.2f} {speedup:>8.2f}%   {outputs_match}"
                )


if __name__ == "__main__":
    main()
