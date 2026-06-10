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

"""Benchmark Llama layer execution with SpectraX scan trace on vs off.

Builds a small Llama-shaped causal LM twice — once with ``scan_layers=True``
(scan-driven traversal) and once with ``scan_layers=False`` (fully unrolled
layer stack) — then JIT-compiles the backbone forward and times steady-state
iterations. The script also monkey-patches SpectraX's ``ModuleList.scan`` /
``StackedModuleList.scan`` to capture the ``trace`` flag actually used at
runtime, so the report shows whether scan-based or unrolled tracing dominated.

Useful for diagnosing the compile-time / iteration-time tradeoff between
scan-on-layers (compact HLO, slightly higher dispatch overhead) and full
unrolling (large HLO, sometimes faster steady-state).

Usage:
    python scripts/bench_llama_scan_trace.py --batch-size 1 --seq-len 128 \\
        --layers 4 --hidden-size 256 --intermediate-size 512
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from spectrax.core.containers import ModuleList, StackedModuleList

import easydel as ed


@dataclass(frozen=True)
class Result:
    """Latency / scan-trace summary for one configuration of the benchmark.

    Attributes:
        name: Human-readable label (``trace_off`` / ``trace_on``).
        scan_layers: The ``scan_layers`` flag the config was built with.
        trace_values: ``trace`` argument seen on each ``ModuleList.scan`` call
            in invocation order during the forward.
        compile_ms: Wall-clock time for the first traced + compiled forward.
        iter_ms: Mean steady-state iteration latency in milliseconds.
        tokens_per_second: Throughput derived from ``batch * seq_len / iter``.
        output_shape: Shape of the resulting ``last_hidden_state``.
    """

    name: str
    scan_layers: bool
    trace_values: tuple[bool, ...]
    compile_ms: float
    iter_ms: float
    tokens_per_second: float
    output_shape: tuple[int, ...]


def _dtype(name: str) -> jnp.dtype:
    """Map a dtype string flag to the corresponding ``jnp.dtype``.

    Args:
        name: One of ``"float32"``, ``"bfloat16"``, ``"float16"``.

    Returns:
        jnp.dtype: The matching JAX dtype.

    Raises:
        ValueError: If ``name`` is not a recognized dtype string.
    """
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float16":
        return jnp.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _block_until_ready(tree: Any) -> Any:
    """Block until every device-array leaf in ``tree`` finishes executing.

    Args:
        tree: Arbitrary pytree whose leaves may include JAX arrays.

    Returns:
        Any: The same pytree, with each blocking-capable leaf replaced by its
            ``block_until_ready`` return value.
    """
    return jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def _build_model(args: argparse.Namespace, *, scan_layers: bool) -> ed.LlamaForCausalLM:
    """Build a small :class:`easydel.LlamaForCausalLM` with the given scan flag.

    Args:
        args: Parsed CLI namespace.
        scan_layers: Whether to scan over decoder layers (vs full unroll).

    Returns:
        ed.LlamaForCausalLM: Freshly initialized causal LM.
    """
    dtype = _dtype(args.dtype)
    cfg = ed.LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=max(args.seq_len, args.max_position_embeddings),
        scan_layers=scan_layers,
        dtype=dtype,
        param_dtype=dtype,
    )
    return ed.LlamaForCausalLM(cfg, rngs=ed.Rngs(args.seed))


def _trace_values(model: ed.LlamaForCausalLM, input_ids: jax.Array) -> tuple[bool, ...]:
    """Capture the ``trace`` flag at every SpectraX scan call during one forward.

    Monkey-patches ``ModuleList.scan`` / ``StackedModuleList.scan`` for the
    duration of a single forward, restoring the originals on exit.

    Args:
        model: Model to invoke with ``input_ids``.
        input_ids: Input tokens for the forward.

    Returns:
        tuple[bool, ...]: ``trace`` flag at each scan call site, in invocation
            order.
    """
    calls: list[bool] = []
    original_module_scan = ModuleList.scan
    original_stacked_scan = StackedModuleList.scan

    def wrapped_scan(self, fn, init_carry, *, trace=False, unroll=None):
        """Record ``trace`` and delegate to the original ``ModuleList.scan``."""
        calls.append(bool(trace))
        original = original_stacked_scan if isinstance(self, StackedModuleList) else original_module_scan
        return original(self, fn, init_carry, trace=trace, unroll=unroll)

    ModuleList.scan = wrapped_scan
    StackedModuleList.scan = wrapped_scan
    try:
        out = model.model(input_ids=input_ids).last_hidden_state
        _block_until_ready(out)
    finally:
        ModuleList.scan = original_module_scan
        StackedModuleList.scan = original_stacked_scan
    return tuple(calls)


def _bench_case(args: argparse.Namespace, *, name: str, scan_layers: bool) -> Result:
    """Build, warm up, and time one scan configuration.

    Builds two models with the same config: the first is used purely to capture
    scan trace flags, the second is the model actually JIT-compiled and timed.

    Args:
        args: Parsed CLI namespace.
        name: Label written into the resulting :class:`Result`.
        scan_layers: ``scan_layers`` flag for this configuration.

    Returns:
        Result: Latency / throughput summary for this configuration.
    """
    input_ids = jnp.ones((args.batch_size, args.seq_len), dtype=jnp.int32)
    trace_values = _trace_values(_build_model(args, scan_layers=scan_layers), input_ids)
    model = _build_model(args, scan_layers=scan_layers)

    @jax.jit
    def run(m, ids):
        """JIT-compiled backbone forward used inside the timing loop."""
        return m.model(input_ids=ids).last_hidden_state

    start = time.perf_counter()
    out = run(model, input_ids)
    _block_until_ready(out)
    compile_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(args.warmup):
        out = run(model, input_ids)
        _block_until_ready(out)

    start = time.perf_counter()
    for _ in range(args.iters):
        out = run(model, input_ids)
        _block_until_ready(out)
    iter_ms = (time.perf_counter() - start) * 1000.0 / args.iters
    tokens_per_second = args.batch_size * args.seq_len / (iter_ms / 1000.0)
    return Result(
        name=name,
        scan_layers=scan_layers,
        trace_values=trace_values,
        compile_ms=compile_ms,
        iter_ms=iter_ms,
        tokens_per_second=tokens_per_second,
        output_shape=tuple(out.shape),
    )


def main() -> None:
    """CLI entry point for the scan-trace benchmark.

    Args:
        --batch-size: Per-step batch size.
        --seq-len: Sequence length for the synthetic forward.
        --layers / --hidden-size / --intermediate-size / --heads / --kv-heads /
            --vocab-size / --max-position-embeddings: Llama architecture knobs.
        --dtype: Storage / compute dtype.
        --warmup / --iters: Iteration counts for warmup / timed runs.
        --seed: Initialization seed.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"backend={jax.default_backend()} devices={len(jax.devices())}")
    print(
        "model="
        f"layers={args.layers} hidden={args.hidden_size} intermediate={args.intermediate_size} "
        f"heads={args.heads} kv_heads={args.kv_heads} batch={args.batch_size} seq={args.seq_len} dtype={args.dtype}"
    )

    trace_off = _bench_case(args, name="trace_off", scan_layers=True)
    trace_on = _bench_case(args, name="trace_on", scan_layers=False)

    for result in (trace_off, trace_on):
        print(
            f"{result.name}: scan_layers={result.scan_layers} trace_values={result.trace_values} "
            f"compile_ms={result.compile_ms:.2f} iter_ms={result.iter_ms:.3f} "
            f"tok/s={result.tokens_per_second:.2f} output_shape={result.output_shape}"
        )

    if trace_off.iter_ms > 0:
        print(f"trace_on/trace_off_iter_ms={trace_on.iter_ms / trace_off.iter_ms:.3f}x")


if __name__ == "__main__":
    main()
