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

"""Benchmark a real 8B-shaped sharded Llama model on TPU/GPU.

Builds a freshly initialized :class:`easydel.LlamaForCausalLM` (or the bare
:class:`easydel.LlamaModel` when ``--base-only`` is passed) with Llama-3-8B
defaults, applies the requested 6D mesh ``(pp, dp, fsdp, ep, tp, sp)``, and
measures parameter sharding, single-step compile time, and steady-state
forward iteration latency / tokens-per-second. The default mesh uses all
visible devices for tensor parallelism: ``(1, 1, 1, 1, -1, 1)``.

Two scan modes can be benchmarked:

* ``trace_off`` — ``scan_layers=True``, layer stack traversed via SpectraX
  ``ModuleList.scan`` without per-step tracing.
* ``trace_on`` — ``scan_layers=False``, every layer is traced unrolled, which
  helps surface XLA fusion behaviour but produces a much larger HLO.

The script also patches ``ModuleList.scan`` / ``StackedModuleList.scan`` to
record the ``trace`` flag each call site uses, so the report reflects what
SpectraX actually did during the timed forward.

Usage:
    python scripts/bench_llama_8b_sharded.py --case both --batch-size 1 \\
        --seq-len 128 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import spectrax as spx
from spectrax.core.containers import ModuleList, StackedModuleList

import easydel as ed

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


@dataclass(frozen=True)
class Result:
    """Single benchmark result captured by :func:`_bench_case`.

    Attributes:
        name: Human-readable label for the case (``trace_off`` / ``trace_on``).
        build_ms: Wall-clock time to construct the model in milliseconds.
        compile_ms: Time spent on the first JIT trace + compile in milliseconds.
        iter_ms: Mean steady-state forward iteration latency in milliseconds.
        tokens_per_second: Throughput derived from ``batch_size * seq_len / iter_ms``.
        output_shape: Shape of the final hidden-state output tensor.
        trace_values: ``trace`` flags observed inside SpectraX scan calls
            during the warmup forward.
        param_count: Total scalar parameter count of the model.
        unique_shardings: Set of distinct partition-spec strings seen across
            parameters.
    """

    name: str
    build_ms: float
    compile_ms: float
    iter_ms: float
    tokens_per_second: float
    output_shape: tuple[int, ...]
    trace_values: tuple[bool, ...]
    param_count: int
    unique_shardings: tuple[str, ...]


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


def _parse_axis_dims(value: str) -> tuple[int, ...]:
    """Parse the ``--axis-dims`` CLI string into a 6-tuple matching :data:`AXIS_NAMES`.

    Args:
        value: Comma-separated integers, e.g. ``"1,1,1,1,-1,1"``. Entries equal
            to ``-1`` request automatic sizing for that axis.

    Returns:
        tuple[int, ...]: Parsed dims, with length equal to ``len(AXIS_NAMES)``.

    Raises:
        ValueError: If the number of entries does not match ``AXIS_NAMES``.
    """
    dims = tuple(int(part.strip()) for part in value.split(","))
    if len(dims) != len(AXIS_NAMES):
        raise ValueError(f"--axis-dims must have {len(AXIS_NAMES)} entries for {AXIS_NAMES}, got {dims}")
    return dims


def _block_until_ready(tree: Any) -> Any:
    """Block until every device-array leaf in ``tree`` finishes executing.

    Args:
        tree: Arbitrary pytree whose leaves may include JAX arrays.

    Returns:
        Any: The same pytree, with each blocking-capable leaf replaced by its
            ``block_until_ready`` return value.
    """
    return jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def _build_config(args: argparse.Namespace, *, scan_layers: bool) -> ed.LlamaConfig:
    """Construct a :class:`easydel.LlamaConfig` from parsed CLI arguments.

    Args:
        args: Parsed CLI namespace produced by :func:`main`.
        scan_layers: If ``True`` configure the model to use SpectraX scan over
            the decoder layer stack; if ``False`` the stack is unrolled.

    Returns:
        easydel.LlamaConfig: A fully populated Llama configuration.
    """
    dtype = _dtype(args.dtype)
    return ed.LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=max(args.seq_len, args.max_position_embeddings),
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        scan_layers=scan_layers,
        dtype=dtype,
        param_dtype=dtype,
        sharding_axis_dims=_parse_axis_dims(args.axis_dims),
        sharding_axis_names=AXIS_NAMES,
        use_sharding_constraint=args.use_sharding_constraint,
    )


def _build_model(args: argparse.Namespace, *, scan_layers: bool) -> ed.LlamaForCausalLM | ed.LlamaModel:
    """Instantiate either :class:`LlamaForCausalLM` or bare :class:`LlamaModel`.

    Args:
        args: Parsed CLI namespace.
        scan_layers: Forwarded into :func:`_build_config`.

    Returns:
        ed.LlamaForCausalLM | ed.LlamaModel: Newly initialized model. Returns
            the bare backbone when ``args.base_only`` is set.
    """
    cfg = _build_config(args, scan_layers=scan_layers)
    dtype = _dtype(args.dtype)
    if args.base_only:
        return ed.LlamaModel(cfg, dtype=dtype, param_dtype=dtype, precision=args.precision, rngs=ed.Rngs(args.seed))
    return ed.LlamaForCausalLM(cfg, dtype=dtype, param_dtype=dtype, precision=args.precision, rngs=ed.Rngs(args.seed))


def _forward(model: ed.LlamaForCausalLM | ed.LlamaModel, input_ids: jax.Array) -> jax.Array:
    """Run a single forward pass and return the final hidden state.

    For full ``ForCausalLM`` models, the LM head is bypassed via
    ``apply_lm_head=False`` so the benchmark measures the backbone cost only.

    Args:
        model: Either a full causal-LM or the bare backbone.
        input_ids: Token ID tensor of shape ``[batch, seq_len]``.

    Returns:
        jax.Array: ``last_hidden_state`` of shape ``[batch, seq_len, hidden]``.
    """
    if hasattr(model, "model"):
        out = model(input_ids=input_ids, apply_lm_head=False)
        return out.last_hidden_state
    return model(input_ids=input_ids).last_hidden_state


def _trace_values(model: ed.LlamaForCausalLM | ed.LlamaModel, input_ids: jax.Array) -> tuple[bool, ...]:
    """Capture the ``trace`` flag at every SpectraX scan call during one forward.

    Monkey-patches ``ModuleList.scan`` and ``StackedModuleList.scan`` for the
    duration of a single forward, restoring the originals on exit. Used to
    confirm what scan mode the model is actually exercising in a given case.

    Args:
        model: Model to run the introspecting forward through.
        input_ids: Input token IDs for the forward pass.

    Returns:
        tuple[bool, ...]: ``trace`` flag value recorded at each scan call site
            in invocation order.
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
        out = _forward(model, input_ids)
        _block_until_ready(out)
    finally:
        ModuleList.scan = original_module_scan
        StackedModuleList.scan = original_stacked_scan
    return tuple(calls)


def _parameter_summary(model: ed.LlamaForCausalLM | ed.LlamaModel) -> tuple[int, tuple[str, ...]]:
    """Compute total parameter count and the set of unique parameter shardings.

    Args:
        model: Model to inspect via ``spx.iter_variables``.

    Returns:
        tuple[int, tuple[str, ...]]: ``(param_count, unique_shardings)`` where
            ``unique_shardings`` is a sorted tuple of partition-spec strings
            (or ``"unsharded"`` for replicated params).
    """
    param_count = 0
    shardings: set[str] = set()
    for _path, var in spx.iter_variables(model):
        if var.kind != "parameters":
            continue
        value = var.value
        if hasattr(value, "size"):
            param_count += int(value.size)
        sharding = getattr(value, "sharding", None)
        spec = getattr(sharding, "spec", None)
        if spec is not None:
            shardings.add(str(spec))
        elif sharding is not None:
            shardings.add(type(sharding).__name__)
        else:
            shardings.add("unsharded")
    return param_count, tuple(sorted(shardings))


def _bench_case(args: argparse.Namespace, *, name: str, scan_layers: bool) -> Result:
    """Build, warm up, and time a single scan configuration.

    Builds a fresh model, captures scan ``trace`` flags, JIT-compiles the
    forward, runs ``args.warmup`` warmup iterations, then times ``args.iters``
    timed iterations to derive steady-state latency / throughput. The model is
    deleted and JAX compile caches are cleared afterwards so subsequent cases
    start from a clean slate.

    Args:
        args: Parsed CLI namespace.
        name: Result label written into the returned :class:`Result`.
        scan_layers: Whether to enable SpectraX layer scan in this case.

    Returns:
        Result: Latency / throughput / sharding summary for this configuration.
    """
    input_ids = jnp.ones((args.batch_size, args.seq_len), dtype=jnp.int32)

    build_start = time.perf_counter()
    model = _build_model(args, scan_layers=scan_layers)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    param_count, unique_shardings = _parameter_summary(model)
    trace_values = _trace_values(model, input_ids)

    @jax.jit
    def run(m, ids):
        """JIT-compiled forward used inside the timing loop."""
        return _forward(m, ids)

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
    iter_ms = (time.perf_counter() - start) * 1000.0 / max(1, args.iters)
    tokens_per_second = args.batch_size * args.seq_len / (iter_ms / 1000.0)

    result = Result(
        name=name,
        build_ms=build_ms,
        compile_ms=compile_ms,
        iter_ms=iter_ms,
        tokens_per_second=tokens_per_second,
        output_shape=tuple(out.shape),
        trace_values=trace_values,
        param_count=param_count,
        unique_shardings=unique_shardings,
    )

    del model, out
    gc.collect()
    jax.clear_caches()
    return result


def _print_result(result: Result) -> None:
    """Pretty-print a :class:`Result` to stdout in two-line summary form.

    Args:
        result: Benchmark output produced by :func:`_bench_case`.

    Returns:
        None.
    """
    print(
        f"{result.name}: trace_values={result.trace_values} "
        f"params={result.param_count:,} unique_shardings={result.unique_shardings}"
    )
    print(
        f"{result.name}: build_ms={result.build_ms:.2f} compile_ms={result.compile_ms:.2f} "
        f"iter_ms={result.iter_ms:.3f} tok/s={result.tokens_per_second:.2f} "
        f"output_shape={result.output_shape}"
    )


def main() -> None:
    """CLI entry point for the sharded 8B Llama benchmark.

    Parses arguments, optionally monkey-patches SpectraX scan's effective
    unroll policy, and runs the selected benchmark cases.

    Args:
        --case: Which configuration to run (``trace_off``, ``trace_on``, or
            ``both``).
        --base-only: When set, benchmark :class:`LlamaModel` instead of
            :class:`LlamaForCausalLM`.
        --batch-size: Per-step batch size.
        --seq-len: Sequence length for the synthetic input.
        --layers / --hidden-size / --intermediate-size / --heads / --kv-heads /
            --vocab-size / --max-position-embeddings / --rope-theta /
            --rms-norm-eps: Llama architecture knobs.
        --dtype: Storage and compute dtype (``bfloat16`` by default).
        --precision: Optional ``jax.lax.Precision`` override.
        --axis-dims: 6-axis mesh dims string consumed by :func:`_parse_axis_dims`.
        --use-sharding-constraint: Toggle SpectraX sharding constraints.
        --warmup / --iters: Iteration counts for warmup / timed runs.
        --seed: Initialization seed.
        --scan-unroll-default: Override SpectraX's default scan unroll factor.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("trace_off", "trace_on", "both"), default="trace_off")
    parser.add_argument(
        "--base-only", action="store_true", help="Benchmark LlamaModel only instead of full ForCausalLM."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--max-position-embeddings", type=int, default=8192)
    parser.add_argument("--rope-theta", type=float, default=500000.0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--axis-dims", default="1,1,1,1,-1,1")
    parser.add_argument("--use-sharding-constraint", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scan-unroll-default",
        type=int,
        default=None,
        help="Override SpectraX scan's unroll=None policy for benchmarking.",
    )
    args = parser.parse_args()

    if args.scan_unroll_default is not None:
        import spectrax.core.containers as containers

        def effective_unroll(unroll, length):
            """Replace SpectraX's default unroll policy with the CLI override."""
            if length <= 0:
                return 1
            if unroll is None:
                return max(1, int(args.scan_unroll_default))
            return max(1, int(unroll))

        containers._scan_effective_unroll = effective_unroll

    print(f"backend={jax.default_backend()} devices={len(jax.devices())}")
    print(f"axis_names={AXIS_NAMES} axis_dims={_parse_axis_dims(args.axis_dims)}")
    print(
        "model="
        f"layers={args.layers} hidden={args.hidden_size} intermediate={args.intermediate_size} "
        f"heads={args.heads} kv_heads={args.kv_heads} vocab={args.vocab_size} "
        f"batch={args.batch_size} seq={args.seq_len} dtype={args.dtype} base_only={args.base_only}"
    )

    if args.case in ("trace_off", "both"):
        _print_result(_bench_case(args, name="trace_off", scan_layers=True))
    if args.case in ("trace_on", "both"):
        _print_result(_bench_case(args, name="trace_on", scan_layers=False))


if __name__ == "__main__":
    main()
