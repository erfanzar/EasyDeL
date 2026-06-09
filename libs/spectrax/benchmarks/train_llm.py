# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end training loop: spectrax vs flax.nnx on a 1.21B transformer.

Runs ``N_EPOCHS`` epochs of ``ITERS_PER_EPOCH`` steps each on deterministic
dummy data, using both libraries with matched hyperparameters (adamw,
same learning rate, same batch/seq/model sizes, bf16 params, bf16
optimizer moments). Times each step with
``jax.block_until_ready`` and reports per-epoch + per-library
summaries at the end.

Usage::

    python -m benchmarks.train_llm --device tpu --model-size 8b
    python -m benchmarks.train_llm --device cpu --n-layers 2 --batch 2

On TPU v5 (1 chip), the defaults fit in HBM when moments are bf16.
Reduce ``--n-layers`` / ``--batch`` for CPU or smaller accelerators.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _set_device(device: str) -> None:
    """Pin the JAX backend before any jax import.

    Must run before any downstream import pulls jax in — the constant
    is only read once at jax initialization time.
    """
    os.environ["JAX_PLATFORMS"] = device
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _median_p05_p95(xs: list[float]) -> tuple[float, float, float]:
    """Return ``(median, p05, p95)`` in the same units as ``xs`` (ms here)."""
    s = sorted(xs)
    n = len(s)

    def pct(p: float) -> float:
        """Interpolate-free percentile — nearest-index of the sorted list."""
        return s[min(n - 1, max(0, int(p * (n - 1))))]

    return s[n // 2], pct(0.05), pct(0.95)


def _bytes_human(n: int) -> str:
    """Format a byte count as KB/MB/GB with one decimal."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def build_spx_train_step(
    n_layers: int,
    batch: int,
    seq_len: int,
    lr: float,
    d_model: int = 2048,
    n_heads: int = 16,
    ffn: int = 8192,
):
    """Build a jit-compiled spectrax training step.

    Uses the native spectrax stack end-to-end — no fallback to raw
    :func:`jax.jit` / :func:`jax.value_and_grad` / hand-rolled optax
    plumbing. This is the symmetric counterpart of the
    :func:`build_nnx_train_step` flax-side builder: both live modules
    (module + optimizer) flow through the jitted step; mutations are
    caught at trace time and propagated back to the originals on return.

    * :func:`spectrax.jit` with ``mutable="params"`` handles the
      module/state split, capture-by-tracer-identity mutation detection,
      and write-back.
    * :func:`spectrax.value_and_grad` differentiates against the same
      ``"params"`` collection the optimizer is sized to.
    * :class:`spectrax.contrib.Optimizer` is a pytree, so it threads
      through the jit call as a normal input/output; its
      :meth:`~spectrax.contrib.Optimizer.apply_eager` method runs the
      optax step and writes the updated params back into the (rebound)
      module inside the trace, which the jit wrapper then mirrors onto
      the live module.

    Returns:
        A 4-tuple ``(step_fn, model, opt, dummy_batch_fn)``:

        * ``step_fn(model, opt, x, y) -> (loss, new_opt)`` — fully
          jittable; the model is mutated in place by the transform,
          and the optimizer is threaded functionally.
        * ``model`` — the spectrax transformer (live reference).
        * ``opt`` — the initial :class:`~spectrax.contrib.Optimizer`,
          sized to ``wrt="params"``.
        * ``dummy_batch_fn(step) -> (x, y)`` — deterministic data.
    """
    import jax
    import jax.numpy as jnp
    import optax

    import spectrax as spx
    import spectrax.contrib as spx_contrib

    from . import models

    model, _x0 = models.spx_transformer_1b(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ffn=ffn,
        batch=batch,
        seq_len=seq_len,
    )
    tx = optax.adamw(lr, mu_dtype=jnp.bfloat16)
    opt = spx_contrib.Optimizer.create(model, tx, wrt="params")

    @spx.jit(mutable="params")
    def step_fn(m, o, x, y):
        """Single train step: forward, MSE loss, grad, optimizer update.

        Under :func:`spectrax.jit`, ``m`` is rebuilt as a tracer-backed
        module on entry and any writes to its :class:`Variable` s are
        captured and applied to the live module on return. The
        :class:`Optimizer` ``o`` is a pytree, so its ``opt_state`` /
        ``step`` flow as normal tracers, and a new :class:`Optimizer`
        comes back out.
        """

        def loss_fn(m):
            """Mean-squared-error regression loss over the full tensor."""
            return ((m(x) - y) ** 2).mean()

        loss, grads = spx.value_and_grad(loss_fn)(m)
        new_opt = o.apply_eager(m, grads)
        return loss, new_opt

    def dummy_batch(step: int):
        """Deterministic ``(x, y)`` batch for step index ``step``."""
        key = jax.random.fold_in(jax.random.PRNGKey(0), step)
        kx, ky = jax.random.split(key)
        x = jax.random.normal(kx, (batch, seq_len, model.blk0.d_model), dtype=jnp.bfloat16)
        y = jax.random.normal(ky, (batch, seq_len, model.blk0.d_model), dtype=jnp.bfloat16)
        return x, y

    return step_fn, model, opt, dummy_batch


def build_nnx_train_step(
    n_layers: int,
    batch: int,
    seq_len: int,
    lr: float,
    d_model: int = 2048,
    n_heads: int = 16,
    ffn: int = 8192,
):
    """Build a jit-compiled flax.nnx training step and initial optimizer state.

    Mirror of :func:`build_spx_train_step`. Uses ``nnx.Optimizer`` with
    ``wrt=nnx.Param`` so optimizer state is scoped to parameters only
    (same scope as the spectrax side's default ``wrt="params"``).
    """
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from . import models

    model, _x0 = models.nnx_transformer_1b(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ffn=ffn,
        batch=batch,
        seq_len=seq_len,
    )
    tx = optax.adamw(lr, mu_dtype=jnp.bfloat16)
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def step_fn(m, o, x, y):
        """Single train step: forward, MSE loss, grad, optimizer update."""

        def loss_fn(m):
            """Mean-squared-error regression loss over the full tensor."""
            return ((m(x) - y) ** 2).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(m)
        o.update(m, grads)
        return loss

    def dummy_batch(step: int):
        """Same RNG fold as the spectrax builder so both see identical data."""
        key = jax.random.fold_in(jax.random.PRNGKey(0), step)
        kx, ky = jax.random.split(key)
        d_model = model.blk0.d_model
        x = jax.random.normal(kx, (batch, seq_len, d_model), dtype=jnp.bfloat16)
        y = jax.random.normal(ky, (batch, seq_len, d_model), dtype=jnp.bfloat16)
        return x, y

    return step_fn, model, opt, dummy_batch


def _run_spx(
    name: str,
    n_epochs: int,
    iters_per_epoch: int,
    n_layers: int,
    batch: int,
    seq_len: int,
    lr: float,
    d_model: int = 2048,
    n_heads: int = 16,
    ffn: int = 8192,
) -> dict:
    """Execute the spectrax training loop and collect per-step timings.

    The model is mutated in place by :func:`spectrax.jit`; the
    :class:`Optimizer` is threaded explicitly as a pytree. Mirror of
    :func:`_run_nnx`.
    """
    import jax

    print(f"\n=== {name} ===")
    step_fn, model, opt, dummy_batch = build_spx_train_step(
        n_layers=n_layers,
        batch=batch,
        seq_len=seq_len,
        lr=lr,
        d_model=d_model,
        n_heads=n_heads,
        ffn=ffn,
    )

    x0, y0 = dummy_batch(0)
    t0 = time.perf_counter_ns()
    loss, opt = step_fn(model, opt, x0, y0)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"  compile + first step: {compile_ms:.1f} ms (loss={float(loss):.4f})")

    epochs: list[dict] = []
    all_steady: list[float] = []
    step_index = 1

    for epoch in range(n_epochs):
        step_times: list[float] = []
        t_epoch = time.perf_counter_ns()
        for _ in range(iters_per_epoch):
            x, y = dummy_batch(step_index)
            t0 = time.perf_counter_ns()
            loss, opt = step_fn(model, opt, x, y)
            jax.block_until_ready(loss)
            step_times.append((time.perf_counter_ns() - t0) / 1e6)
            step_index += 1
        epoch_total_s = (time.perf_counter_ns() - t_epoch) / 1e9
        med, p05, p95 = _median_p05_p95(step_times)
        epochs.append({"median_ms": med, "p05_ms": p05, "p95_ms": p95, "total_s": epoch_total_s})
        all_steady.extend(step_times)
        print(
            f"  epoch {epoch + 1}/{n_epochs}: "
            f"median={med:.2f} ms  p05={p05:.2f}  p95={p95:.2f}  "
            f"loss={float(loss):.4f}  epoch_total={epoch_total_s:.2f} s"
        )

    return {
        "name": name,
        "compile_ms": compile_ms,
        "epochs": epochs,
        "all_steady": all_steady,
        "final_loss": float(loss),
    }


def _run_nnx(
    name: str,
    n_epochs: int,
    iters_per_epoch: int,
    n_layers: int,
    batch: int,
    seq_len: int,
    lr: float,
    d_model: int = 2048,
    n_heads: int = 16,
    ffn: int = 8192,
) -> dict:
    """Execute the flax.nnx training loop — mirror of :func:`_run_spx`.

    :class:`nnx.Optimizer` is a pytree-registered object, so the
    module and optimizer flow through :func:`nnx.jit` directly and
    are mutated in place inside the step.
    """
    import jax

    print(f"\n=== {name} ===")
    step_fn, model, opt, dummy_batch = build_nnx_train_step(
        n_layers=n_layers,
        batch=batch,
        seq_len=seq_len,
        lr=lr,
        d_model=d_model,
        n_heads=n_heads,
        ffn=ffn,
    )

    x0, y0 = dummy_batch(0)
    t0 = time.perf_counter_ns()
    loss = step_fn(model, opt, x0, y0)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"  compile + first step: {compile_ms:.1f} ms (loss={float(loss):.4f})")

    epochs: list[dict] = []
    all_steady: list[float] = []
    step_index = 1

    for epoch in range(n_epochs):
        step_times: list[float] = []
        t_epoch = time.perf_counter_ns()
        for _ in range(iters_per_epoch):
            x, y = dummy_batch(step_index)
            t0 = time.perf_counter_ns()
            loss = step_fn(model, opt, x, y)
            jax.block_until_ready(loss)
            step_times.append((time.perf_counter_ns() - t0) / 1e6)
            step_index += 1
        epoch_total_s = (time.perf_counter_ns() - t_epoch) / 1e9
        med, p05, p95 = _median_p05_p95(step_times)
        epochs.append({"median_ms": med, "p05_ms": p05, "p95_ms": p95, "total_s": epoch_total_s})
        all_steady.extend(step_times)
        print(
            f"  epoch {epoch + 1}/{n_epochs}: "
            f"median={med:.2f} ms  p05={p05:.2f}  p95={p95:.2f}  "
            f"loss={float(loss):.4f}  epoch_total={epoch_total_s:.2f} s"
        )

    return {
        "name": name,
        "compile_ms": compile_ms,
        "epochs": epochs,
        "all_steady": all_steady,
        "final_loss": float(loss),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — parse args, run both libraries, emit a summary.

    The two libraries run sequentially (not in parallel) so they do not
    contend for accelerator memory. The returned exit code is always
    zero; any failure surfaces as an exception.
    """
    parser = argparse.ArgumentParser(description="1B transformer: spectrax vs flax.nnx training comparison")
    parser.add_argument("--device", default="tpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100, help="steps per epoch")
    parser.add_argument(
        "--model-size",
        choices=["1b", "3b", "8b", "custom"],
        default="1b",
        help="preset shape; 'custom' forces using --n-layers / --d-model / --n-heads / --ffn explicitly",
    )
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="benchmarks/results/train_llm.json")
    parser.add_argument("--only", choices=["spx", "nnx", "both"], default="both")
    args = parser.parse_args(argv)

    presets = {
        "1b": {"n_layers": 24, "d_model": 2048, "n_heads": 16, "ffn": 8192, "batch": 4, "seq_len": 512},
        "3b": {"n_layers": 32, "d_model": 3072, "n_heads": 24, "ffn": 12288, "batch": 2, "seq_len": 512},
        "8b": {"n_layers": 40, "d_model": 4096, "n_heads": 32, "ffn": 16384, "batch": 2, "seq_len": 512},
        "custom": {},
    }
    preset = presets[args.model_size]
    args.n_layers = args.n_layers if args.n_layers is not None else preset.get("n_layers", 24)
    args.d_model = args.d_model if args.d_model is not None else preset.get("d_model", 2048)
    args.n_heads = args.n_heads if args.n_heads is not None else preset.get("n_heads", 16)
    args.ffn = args.ffn if args.ffn is not None else preset.get("ffn", 8192)
    args.batch = args.batch if args.batch is not None else preset.get("batch", 4)
    args.seq_len = args.seq_len if args.seq_len is not None else preset.get("seq_len", 512)
    total_params = args.n_layers * (4 * args.d_model * args.d_model + 2 * args.d_model * args.ffn)
    args.total_params = total_params

    _set_device(args.device)

    import jax

    print("spectrax vs flax.nnx — transformer training run")
    print(f"  device        : {args.device} ({jax.devices()})")
    print(f"  model size    : {args.model_size}  (~{args.total_params / 1e9:.2f}B params)")
    print(f"  n_layers      : {args.n_layers}")
    print(f"  d_model/heads : {args.d_model} / {args.n_heads}")
    print(f"  ffn           : {args.ffn}")
    print(f"  batch         : {args.batch}")
    print(f"  seq_len       : {args.seq_len}")
    print(f"  lr            : {args.lr}")
    print(f"  epochs        : {args.epochs} x {args.iters} iters = {args.epochs * args.iters} steps")

    results: dict[str, dict] = {}
    if args.only in ("spx", "both"):
        results["spectrax"] = _run_spx(
            "spectrax",
            args.epochs,
            args.iters,
            args.n_layers,
            args.batch,
            args.seq_len,
            args.lr,
            d_model=args.d_model,
            n_heads=args.n_heads,
            ffn=args.ffn,
        )
    if args.only in ("nnx", "both"):
        results["nnx"] = _run_nnx(
            "nnx",
            args.epochs,
            args.iters,
            args.n_layers,
            args.batch,
            args.seq_len,
            args.lr,
            d_model=args.d_model,
            n_heads=args.n_heads,
            ffn=args.ffn,
        )

    print("\n=== final summary ===")
    print(
        f"{'library':<10} {'compile ms':>11} {'steady med':>11} {'steady p05':>11} {'steady p95':>11} {'total steady (s)':>18} {'final loss':>12}"
    )
    for lib, r in results.items():
        med, p05, p95 = _median_p05_p95(r["all_steady"])
        total_s = sum(r["all_steady"]) / 1000.0
        print(
            f"{lib:<10} {r['compile_ms']:>11.1f} {med:>11.2f} {p05:>11.2f} {p95:>11.2f} {total_s:>18.2f} {r['final_loss']:>12.4f}"
        )

    if "spectrax" in results and "nnx" in results:
        spx_med, _, _ = _median_p05_p95(results["spectrax"]["all_steady"])
        nnx_med, _, _ = _median_p05_p95(results["nnx"]["all_steady"])
        ratio = spx_med / max(nnx_med, 1e-9)
        verdict = "FASTER" if ratio < 1.0 else "SLOWER"
        magnitude = (1.0 / ratio) if ratio < 1.0 else ratio
        print(
            f"\nspectrax is {magnitude:.2f}x {verdict} than nnx on this workload "
            f"(median: {spx_med:.2f} ms vs {nnx_med:.2f} ms; ratio={ratio:.3f})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "args": vars(args),
        "device": args.device,
        "results": {
            lib: {k: v for k, v in r.items() if k != "all_steady"} | {"all_steady_ms": r["all_steady"]}
            for lib, r in results.items()
        },
    }
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nresults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
