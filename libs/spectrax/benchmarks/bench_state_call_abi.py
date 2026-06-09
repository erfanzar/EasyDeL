"""Microbenchmark State call-boundary overhead for ``jax.jit`` vs ``spx.jit``.

This benchmark intentionally uses a tiny compiled body and many State leaves so
the measurement is dominated by Python/JAX dispatch-time pytree handling rather
than device compute. It answers whether SpectraX's automatic State call ABI
matches the hand-written "cache state leaves and pass a tuple" serving pattern.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp

import spectrax as spx


def _make_state(num_leaves: int) -> spx.State:
    return spx.State({"parameters": {f"w{i:05d}": jnp.asarray(float(i), dtype=jnp.float32) for i in range(num_leaves)}})


def _block(value: object) -> None:
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)


def _bench(name: str, call: Callable[[], object], *, warmup: int, iters: int, repeats: int) -> tuple[str, float, float]:
    for _ in range(warmup):
        _block(call())

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _i in range(iters):
            _block(call())
        samples.append((time.perf_counter() - start) / iters)

    median = statistics.median(samples)
    spread = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    print(f"{name:<28} {median * 1e6:>10.2f} us/call  {1.0 / median:>10.1f} calls/s  stdev={spread * 1e6:.2f} us")
    return name, median, spread


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaves", type=int, default=512)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    state = _make_state(args.leaves)
    abi = state.call_abi()
    state_leaves = abi.flatten(state)
    x = jnp.asarray(1.0, dtype=jnp.float32)

    @jax.jit
    def plain_jax_state(model_state: spx.State, value):
        return model_state.get("parameters", "w00000") + value

    @spx.jit
    def auto_spx_state(model_state: spx.State, value):
        return model_state.get("parameters", "w00000") + value

    @jax.jit
    def manual_abi(flat_state: tuple[object, ...], value):
        model_state = abi.unflatten(flat_state)
        return model_state.get("parameters", "w00000") + value

    @jax.jit
    def plain_tuple(flat_state: tuple[object, ...], value):
        return flat_state[0] + value

    _block(plain_jax_state(state, x))
    _block(auto_spx_state(state, x))
    _block(manual_abi(state_leaves, x))
    _block(plain_tuple(state_leaves, x))

    print(f"device={jax.default_backend()} leaves={args.leaves} iters={args.iters} repeats={args.repeats}")
    results = [
        _bench(
            "plain jax.jit(State)",
            lambda: plain_jax_state(state, x),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        ),
        _bench(
            "spx.jit auto(State)",
            lambda: auto_spx_state(state, x),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        ),
        _bench(
            "manual ABI cached leaves",
            lambda: manual_abi(state_leaves, x),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        ),
        _bench(
            "plain tuple cached leaves",
            lambda: plain_tuple(state_leaves, x),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        ),
        _bench(
            "manual ABI flatten/call",
            lambda: manual_abi(abi.flatten(state), x),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        ),
    ]

    timings = {name: median for name, median, _spread in results}
    base = timings["manual ABI cached leaves"]
    auto = timings["spx.jit auto(State)"]
    print(f"\nspx auto / manual cached = {auto / base:.2f}x latency")


if __name__ == "__main__":
    main()
