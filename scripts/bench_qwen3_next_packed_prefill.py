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

"""Benchmark Qwen3Next packed prefill legacy vs refactored helpers.

Times two implementations of the Qwen3Next packed-update kernel against each
other on a synthetic packed-prefill workload:

* :func:`_apply_qwen3_next_packed_updates_legacy` — the original
  per-request loop that calls the gated delta-rule decode kernel inside a
  Python-level scan.
* :func:`_apply_qwen3_next_packed_updates` — the refactored single-call path
  that does the same work using packed tensor ops.

For each combination of synthetic schedule shape (``decode_like``, ``mixed``,
``prefill_heavy``) and token bucket (``512``, ``2048``), the script measures
mean latency and reports the unified path's speedup percentage along with an
``allclose`` cross-check on the outputs. Logical mesh layout (``fsdp4`` or
``tp4``) and the grouped-decode backend (``auto``, ``jax``, ``pallas``) are
selectable via CLI.

Usage:
    python scripts/bench_qwen3_next_packed_prefill.py --layout tp4 \\
        --gdr-backend auto --warmup 2 --repeats 5
"""

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
    """Force :class:`GatedDeltaRuleOp` to use the pure-JAX grouped decode path.

    Bound via ``MethodType`` to override ``GatedDeltaRuleOp.grouped_gdr_decode``
    when ``--gdr-backend jax`` is selected.

    Args:
        self: The :class:`GatedDeltaRuleOp` instance.
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        beta: Per-step gating scalar tensor.
        decay: Per-step decay tensor.
        recurrent_state: Carried recurrent state.

    Returns:
        Result of :meth:`GatedDeltaRuleOp.grouped_gdr_decode_jax`.
    """
    return GatedDeltaRuleOp.grouped_gdr_decode_jax(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
    )


def _grouped_gdr_decode_pallas_only(self, query, key, value, beta, decay, recurrent_state):
    """Force :class:`GatedDeltaRuleOp` to use the Pallas grouped decode path.

    Bound via ``MethodType`` to override ``GatedDeltaRuleOp.grouped_gdr_decode``
    when ``--gdr-backend pallas`` is selected.

    Args:
        self: The :class:`GatedDeltaRuleOp` instance.
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        beta: Per-step gating scalar tensor.
        decay: Per-step decay tensor.
        recurrent_state: Carried recurrent state.

    Returns:
        Result of :meth:`GatedDeltaRuleOp.grouped_gdr_decode_shard_map_pallas`.
    """
    return self.grouped_gdr_decode_shard_map_pallas(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
    )


def _make_gdr_op(layout: str, runtime_dtype=jnp.bfloat16, grouped_decode_backend: str = "auto"):
    """Construct a configured :class:`GatedDeltaRuleOp` for the benchmark.

    Builds a :class:`Qwen3NextConfig` with the chosen mesh layout, wraps it in
    an :class:`OperationMetadata`, and optionally pins the grouped-decode
    backend to either the pure-JAX or Pallas implementation.

    Args:
        layout: Mesh layout key from :data:`LAYOUT_AXIS_DIMS` (``"fsdp4"`` or
            ``"tp4"``).
        runtime_dtype: Runtime dtype for the op (default ``bfloat16``).
        grouped_decode_backend: ``"auto"``, ``"jax"``, or ``"pallas"``. The
            non-auto values monkey-patch ``grouped_gdr_decode`` on the
            returned op.

    Returns:
        GatedDeltaRuleOp: Operation instance configured for the benchmark.
    """
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
    """Build a synthetic packed schedule for the requested workload shape.

    Three shapes are supported:

    * ``decode_like`` — up to 32 requests, each contributing one token (the
      "pure decode" extreme).
    * ``mixed`` — about a quarter of the requests are single-token decodes
      and the rest are multi-token chunks that approximately fill ``bucket``.
    * ``prefill_heavy`` — every request contributes ``bucket / num_requests``
      tokens (the prefill-dominated extreme).

    Args:
        case: One of ``"decode_like"``, ``"mixed"``, ``"prefill_heavy"``.
        bucket: Target packed token budget.
        num_slots: Number of physical request slots.

    Returns:
        tuple[np.ndarray, int]: ``(query_start_loc, num_requests)``. The
            ``query_start_loc`` array has shape ``[num_slots + 1]`` with
            inactive slots padded by repeating the last active offset.

    Raises:
        ValueError: If ``case`` is not a recognized workload shape.
    """
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
    """Generate synthetic tensors and shape metadata for one benchmark case.

    Allocates conv / recurrent state tensors per slot, conv input / beta /
    decay tensors over the packed bucket, and a conv kernel; together these
    feed both ``_apply_qwen3_next_packed_updates_legacy`` and the unified
    helper without further reshaping.

    Args:
        case: Workload shape forwarded to :func:`_build_schedule`.
        bucket: Packed token budget.
        num_slots: Number of physical request slots.
        dtype: Storage dtype for state / activation tensors.

    Returns:
        dict[str, object]: Keyword-argument dictionary expected by both packed
            update helpers, including state tensors, packed activations, the
            per-slot ``query_start_loc``, head dims, and ``expand_ratio``.
    """
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
    """Block until every device-array leaf in ``tree`` finishes executing.

    Args:
        tree: Arbitrary pytree.

    Returns:
        Any: Same pytree with blocking-capable leaves resolved.
    """
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def _time_callable(fn, *, warmup: int, repeats: int) -> tuple[object, float]:
    """Warm up, then time a callable over ``repeats`` iterations.

    Blocks on the output between iterations so the measured time reflects
    device-side execution rather than dispatch only.

    Args:
        fn: Zero-argument callable returning a pytree of arrays.
        warmup: Number of unmeasured warmup iterations.
        repeats: Number of timed iterations to average over.

    Returns:
        tuple[object, float]: The last output produced by ``fn`` and the mean
            iteration time in milliseconds.
    """
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
    """Compare two array pytrees leaf-wise with ``jnp.allclose`` in float32.

    Args:
        lhs: First pytree of arrays.
        rhs: Second pytree of arrays. Must have the same structure as ``lhs``.
        rtol: Relative tolerance for :func:`jnp.allclose`.
        atol: Absolute tolerance for :func:`jnp.allclose`.

    Returns:
        bool: ``True`` if every paired leaf is close within tolerance.
    """
    leaves = zip(jax.tree_util.tree_leaves(lhs), jax.tree_util.tree_leaves(rhs), strict=True)
    return all(jnp.allclose(a.astype(jnp.float32), b.astype(jnp.float32), rtol=rtol, atol=atol) for a, b in leaves)


def main() -> None:
    """CLI entry point for the Qwen3Next packed-prefill benchmark.

    Args:
        --warmup: Warmup iterations per benchmark case.
        --repeats: Timed iterations per case.
        --num-slots: Packed slot count.
        --layout: Mesh layout key (``fsdp4`` or ``tp4``).
        --gdr-backend: Grouped decode backend (``auto``, ``jax``, ``pallas``).

    Returns:
        None. Prints a fixed-width table of per-case latency and the
        ``outputs_match`` cross-check result to stdout.

    Raises:
        RuntimeError: If the constructed :class:`GatedDeltaRuleOp` lacks a
            mesh in its metadata.
    """
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
    if mesh is None:
        raise RuntimeError("GatedDeltaRuleOp metadata did not provide a mesh.")

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
