# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Baseline + verify bench for the fused MLP block.

Measures the full MLP (``down(act(gate(x)) * up(x))``) rather than single
matmuls, in the regimes that matter:

* decode fwd (m=8/32) — inference serving;
* prefill fwd (m=2048) — inference prefill;
* train fwd+bwd (m=2048, ``jax.grad`` through the block) — training step cost.

Measurement rules learned the hard way (see project notes): everything jitted;
in-graph per-op numbers come from a chain of four *distinct* MLPs whose
outputs are fully consumed (XLA dead-column-eliminates sliced outputs); no
chaining operand may be weight-sized.

Sections print ``PATH=<name> ...`` rows; once the fused op lands it appears as
additional paths in the same table for before/after comparison.
"""

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

SHAPES = [
    ("13B", 5120, 13824),
    ("27B", 8192, 28672),
]
DECODE_TOKENS = (8, 32)
TRAIN_TOKENS = 2048
WARMUP = 3
ITERS = 12
CHAIN = 4


def timed(fn, *args):
    """Median step seconds of a compiled call."""
    jax.block_until_ready(fn(*args))
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(ITERS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def mlp_bf16(x, w_gate, w_up, w_down):
    """Reference bf16 MLP block (unfused ops; XLA schedules them)."""
    return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down


def consume(out, k_dim):
    """Fold an [m, K] output back to [m, K] consuming every element."""
    return jnp.tanh(out)


def main():
    """Print the baseline table."""
    print(f"device: {jax.devices()[0].device_kind} warmup={WARMUP} iters={ITERS} chain={CHAIN}", flush=True)
    rng = np.random.default_rng(0)

    for label, k_dim, i_dim in SHAPES:
        weight_sets = []
        for _ in range(CHAIN):
            weight_sets.append(
                tuple(
                    jnp.asarray(rng.normal(size=shape) * 0.02, jnp.bfloat16)
                    for shape in ((k_dim, i_dim), (k_dim, i_dim), (i_dim, k_dim))
                )
            )
        w_gate, w_up, w_down = weight_sets[0]
        total_mb = sum(w.nbytes for w in weight_sets[0]) / 1e6
        print(f"\n=== {label} MLP K={k_dim} I={i_dim} (bf16 weights {total_mb:.0f}MB) ===", flush=True)

        # --- decode fwd, single dispatch and in-graph ---
        for tokens in DECODE_TOKENS:
            x = jnp.asarray(rng.normal(size=(tokens, k_dim)), jnp.bfloat16)
            single = timed(jax.jit(mlp_bf16), x, w_gate, w_up, w_down)

            def chain_fn(x, ws=weight_sets):
                h = x
                for wg, wu, wd in ws:
                    h = consume(mlp_bf16(h, wg, wu, wd), k_dim)
                return h

            in_graph = timed(jax.jit(chain_fn), x) / CHAIN
            print(
                f"PATH=bf16-fwd      t={tokens:<5d} single={single * 1e6:8.1f}us  in-graph/op={in_graph * 1e6:8.1f}us",
                flush=True,
            )

        # --- prefill fwd ---
        x = jnp.asarray(rng.normal(size=(TRAIN_TOKENS, k_dim)), jnp.bfloat16)
        single = timed(jax.jit(mlp_bf16), x, w_gate, w_up, w_down)
        print(f"PATH=bf16-fwd      t={TRAIN_TOKENS:<5d} single={single * 1e6:8.1f}us", flush=True)

        # --- train fwd+bwd ---
        def loss(params, x):
            wg, wu, wd = params
            return jnp.sum(mlp_bf16(x, wg, wu, wd).astype(jnp.float32) ** 2)

        grad_fn = jax.jit(jax.grad(loss))
        train = timed(grad_fn, (w_gate, w_up, w_down), x)
        print(f"PATH=bf16-fwd+bwd  t={TRAIN_TOKENS:<5d} single={train * 1e6:8.1f}us", flush=True)

        # --- quantized decode baseline: unfused w4a4 gemv per projection ---
        try:
            from ejkernel.kernels._pallas.tpu.quantized_matmul._packed_gemv import (
                pack_int4_adjacent,
                w4a4_gemv,
            )

            def q4(w):
                dense = np.asarray(w, np.float32)
                scale = np.abs(dense).max(axis=0, keepdims=True) / 7.0
                codes = np.clip(np.round(dense / scale), -7, 7).astype(np.int32)
                return pack_int4_adjacent(jnp.asarray(codes)), jnp.asarray(scale, jnp.float32)

            gate_up_packed, gate_up_scale = q4(jnp.concatenate([w_gate, w_up], axis=1))
            down_packed, down_scale = q4(w_down)

            def unfused_q_mlp(x):
                x4 = jnp.clip(jnp.round(x.astype(jnp.float32) / 0.25), -7, 7).astype(jnp.int4)
                gate_up = w4a4_gemv(x4, gate_up_packed, gate_up_scale, tile_n=1024) * 0.25
                h = jax.nn.silu(gate_up[:, :i_dim]) * gate_up[:, i_dim:]
                h4 = jnp.clip(jnp.round(h.astype(jnp.float32) / 0.25), -7, 7).astype(jnp.int4)
                pad = (-k_dim) % 1024
                out = w4a4_gemv(h4, down_packed, down_scale, tile_n=512 if k_dim % 1024 else 1024) * 0.25
                return out[:, :k_dim] if pad else out

            for tokens in DECODE_TOKENS:
                x = jnp.asarray(rng.normal(size=(tokens, k_dim)), jnp.bfloat16)
                single = timed(jax.jit(unfused_q_mlp), x)
                print(f"PATH=w4a4-unfused  t={tokens:<5d} single={single * 1e6:8.1f}us", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"PATH=w4a4-unfused  SKIPPED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)

        # --- fused single-dispatch W4A4 MLP (the new kernel) ---
        try:
            from ejkernel.modules import fused_mlp as fused_mlp_op

            def q4_sep(w):
                dense = np.asarray(w, np.float32)
                scale = np.abs(dense).max(axis=0, keepdims=True) / 7.0
                codes = np.clip(np.round(dense / scale), -7, 7).astype(np.int32)
                return pack_int4_adjacent(jnp.asarray(codes)), jnp.asarray(scale, jnp.float32)

            gate_packed, gate_sc = q4_sep(w_gate)
            up_packed, up_sc = q4_sep(w_up)
            down_packed, down_sc = q4_sep(w_down)
            dummy_codes = tuple(
                jnp.zeros((shape), jnp.int4)
                for shape in ((k_dim, i_dim), (k_dim, i_dim), (i_dim, k_dim))
            )

            def fused_call(x):
                return fused_mlp_op(
                    x,
                    *dummy_codes,
                    gate_scale=gate_sc,
                    up_scale=up_sc,
                    down_scale=down_sc,
                    packed_weights=(gate_packed, up_packed, down_packed),
                    packed_tile_i=512,
                )

            for tokens in DECODE_TOKENS:
                x = jnp.asarray(rng.normal(size=(tokens, k_dim)), jnp.bfloat16)
                single = timed(jax.jit(fused_call), x)
                print(f"PATH=w4a4-FUSED    t={tokens:<5d} single={single * 1e6:8.1f}us", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"PATH=w4a4-FUSED    SKIPPED: {type(exc).__name__}: {str(exc)[:140]}", flush=True)

        # --- quantized (frozen) training step through the differentiable path ---
        try:
            from ejkernel.modules import fused_mlp as fused_mlp_op

            def qc(w):
                dense = np.asarray(w, np.float32)
                scale = np.abs(dense).max(axis=0, keepdims=True) / 127.0
                codes = np.clip(np.round(dense / scale), -127, 127).astype(np.int8)
                return jnp.asarray(codes), jnp.asarray(scale, jnp.float32)

            (gq, gs), (uq, us), (dq, ds) = qc(w_gate), qc(w_up), qc(w_down)
            x = jnp.asarray(rng.normal(size=(TRAIN_TOKENS, k_dim)), jnp.bfloat16)

            def frozen_loss(x):
                out = fused_mlp_op(
                    x, gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds,
                    quantize_activations=True,
                )
                return jnp.sum(out.astype(jnp.float32) ** 2)

            train_q = timed(jax.jit(jax.grad(frozen_loss)), x)
            print(f"PATH=int8-frozen fwd+bwd t={TRAIN_TOKENS:<5d} single={train_q * 1e6:8.1f}us", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"PATH=int8-frozen fwd+bwd SKIPPED: {type(exc).__name__}: {str(exc)[:140]}", flush=True)


if __name__ == "__main__":
    main()
