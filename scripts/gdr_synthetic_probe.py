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

"""Synthetic probes for the packed grouped GDR TPU Pallas path.

Exercises the exact dispatch hit by Qwen3.5/3.6 packed training:
query_heads != value_heads, seg_ids is not None, use_chunked=True.

Modes:
  interp   CPU-only numerical check (Pallas interpret mode) against the XLA
           recurrent fallback. Safe to run on a busy TPU host.
  tiny     Level-1 TPU probe: B=8 L=1024 Hq=4 Hv=12 K=V=128 chunk=128,
           fwd + value_and_grad, plus on-device check vs recurrent fallback.
  compile  Level-2 TPU probe: per-TP-shard head shapes (dp2/fsdp64/tp4 shard
           of Qwen3.5: Hq=16/4=4, Hv=48/4=12), fwd or grad, reports compile
           time, step time, and HBM.

TPU modes shard the batch over all local devices with shard_map (batch must
divide the device count — pick a shardable batch size), mirroring how the
trainer's GDR op runs under shard_map instead of a single-chip toy. Pass
--no-shard to force single-device execution.

Examples:
  python scripts/gdr_synthetic_probe.py interp
  python scripts/gdr_synthetic_probe.py tiny
  python scripts/gdr_synthetic_probe.py compile --batch 1 --seqlen 131072 --grad
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["interp", "tiny", "compile"])
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seqlen", type=int, default=None)
    p.add_argument("--hq", type=int, default=None)
    p.add_argument("--hv", type=int, default=None)
    p.add_argument("--dimk", type=int, default=128)
    p.add_argument("--dimv", type=int, default=128)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--seg-period", type=int, default=None, help="tokens per packed document; 0 disables seg_ids")
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16"])
    p.add_argument("--grad", action="store_true", help="also run value_and_grad")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--no-shard", action="store_true", help="run on a single device instead of shard_map over the local mesh")
    return p.parse_args(argv)


def _defaults(args):
    if args.mode == "interp":
        args.batch = args.batch or 2
        args.seqlen = args.seqlen or 512
        args.hq = args.hq or 2
        args.hv = args.hv or 6
        args.dimk = min(args.dimk, 32) if args.dimk == 128 else args.dimk
        args.dimv = min(args.dimv, 32) if args.dimv == 128 else args.dimv
        args.seg_period = 128 if args.seg_period is None else args.seg_period
        args.dtype = args.dtype or "float32"
        args.grad = True
    elif args.mode == "tiny":
        args.batch = args.batch or 8
        args.seqlen = args.seqlen or 1024
        args.hq = args.hq or 4
        args.hv = args.hv or 12
        args.seg_period = 256 if args.seg_period is None else args.seg_period
        args.dtype = args.dtype or "bfloat16"
        args.grad = True
    else:  # compile: per-TP-shard heads of the real run (Hq=16/Hv=48 -> 4/12, ratio 3)
        args.batch = args.batch or 8
        args.seqlen = args.seqlen or 8192
        args.hq = args.hq or 4
        args.hv = args.hv or 12
        args.seg_period = 4096 if args.seg_period is None else args.seg_period
        args.dtype = args.dtype or "bfloat16"
    return args


def main(argv=None):
    args = _defaults(_parse_args(argv if argv is not None else sys.argv[1:]))
    if args.mode == "interp":
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp

    from ejkernel.kernels._pallas.tpu.gated_delta_rule._interface import gated_delta_rule as gdr

    dtype = jnp.dtype(args.dtype)
    B, L, Hq, Hv, K, V = args.batch, args.seqlen, args.hq, args.hv, args.dimk, args.dimv
    print(
        f"[probe:{args.mode}] B={B} L={L} Hq={Hq} Hv={Hv} K={K} V={V} "
        f"chunk={args.chunk} seg_period={args.seg_period} dtype={dtype.name} "
        f"devices={jax.devices()}"
    )

    ks = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(ks[0], (B, L, Hq, K), dtype)
    k = jax.random.normal(ks[1], (B, L, Hq, K), dtype)
    v = jax.random.normal(ks[2], (B, L, Hv, V), dtype)
    beta = jax.nn.sigmoid(jax.random.normal(ks[3], (B, L, Hv), dtype))
    decay = -jax.nn.softplus(jax.random.normal(ks[4], (B, L, Hv), dtype))
    ops = (q, k, v, beta, decay)
    if args.seg_period:
        ops = ops + (jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32) // args.seg_period, (B, L)),)

    def _fwd(use_chunked):
        def fn(*ops):
            seg = ops[5] if len(ops) == 6 else None
            return gdr(*ops[:5], chunk_size=args.chunk, seg_ids=seg, use_chunked=use_chunked)

        return fn

    fwd_chunked, fwd_recurrent = _fwd(True), _fwd(False)

    if args.mode != "interp" and not args.no_shard and jax.local_device_count() > 1:
        import numpy as np
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        try:
            from jax import shard_map
        except ImportError:
            from jax.experimental.shard_map import shard_map

        ndev = jax.local_device_count()
        if B % ndev:
            raise SystemExit(f"batch {B} does not divide {ndev} local devices; pick a shardable batch or pass --no-shard")
        mesh = Mesh(np.asarray(jax.local_devices()), ("dp",))
        in_specs = tuple(P("dp") for _ in ops)
        out_specs = (P("dp"), P("dp"))
        fwd_chunked = shard_map(fwd_chunked, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
        fwd_recurrent = shard_map(fwd_recurrent, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
        ops = tuple(jax.device_put(x, NamedSharding(mesh, P("dp"))) for x in ops)
        print(f"[probe] shard_map over {ndev} local devices: per-device batch {B // ndev}")

    def loss_chunked(*ops):
        out, _ = fwd_chunked(*ops)
        return jnp.mean(jnp.square(out.astype(jnp.float32)))

    def loss_recurrent(*ops):
        out, _ = fwd_recurrent(*ops)
        return jnp.mean(jnp.square(out.astype(jnp.float32)))

    def maxdiff(a, b):
        return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))

    failed = False

    if args.mode == "interp":
        from jax.experimental.pallas import tpu as pltpu

        with pltpu.force_tpu_interpret_mode():
            out_p, st_p = fwd_chunked(*ops)
        out_r, st_r = fwd_recurrent(*ops)
        d_out, d_st = maxdiff(out_p, out_r), maxdiff(st_p, st_r)
        tol = 1e-2 if dtype == jnp.float32 else 6e-2
        print(f"[interp] fwd  max|out-ref|={d_out:.3e} max|state-ref|={d_st:.3e} tol={tol:.0e}")
        failed |= not (d_out < tol and d_st < tol)

        with pltpu.force_tpu_interpret_mode():
            val_p, grads_p = jax.value_and_grad(loss_chunked, argnums=(0, 1, 2, 3, 4))(*ops)
        val_r, grads_r = jax.value_and_grad(loss_recurrent, argnums=(0, 1, 2, 3, 4))(*ops)
        print(f"[interp] loss pallas={float(val_p):.6f} recurrent={float(val_r):.6f}")
        for name, gp, gr in zip(("dq", "dk", "dv", "dbeta", "ddecay"), grads_p, grads_r):
            scale = float(jnp.maximum(jnp.max(jnp.abs(gr.astype(jnp.float32))), 1e-6))
            rel = maxdiff(gp, gr) / scale
            print(f"[interp] grad {name:6s} max|diff|/max|ref|={rel:.3e}")
            failed |= not (rel < 5e-2)
        print("[interp]", "FAIL" if failed else "PASS")
        return 1 if failed else 0

    # --- TPU modes ---
    fwd_j = jax.jit(fwd_chunked)
    t0 = time.perf_counter()
    out, st = jax.block_until_ready(fwd_j(*ops))
    t_compile = time.perf_counter() - t0
    print(f"[{args.mode}] fwd compile+run {t_compile:.2f}s out={out.shape} state={st.shape}")
    print(f"[{args.mode}] fwd finite: out={bool(jnp.all(jnp.isfinite(out.astype(jnp.float32))))} state={bool(jnp.all(jnp.isfinite(st.astype(jnp.float32))))}")
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fwd_j(*ops))
        times.append(time.perf_counter() - t0)
    print(f"[{args.mode}] fwd step {min(times) * 1e3:.1f}ms (best of {args.iters})")

    if args.grad:
        vag = jax.jit(jax.value_and_grad(loss_chunked, argnums=(0, 1, 2, 3, 4)))
        t0 = time.perf_counter()
        val, grads = jax.block_until_ready(vag(*ops))
        t_compile = time.perf_counter() - t0
        gfin = all(bool(jnp.all(jnp.isfinite(g.astype(jnp.float32)))) for g in grads)
        print(f"[{args.mode}] grad compile+run {t_compile:.2f}s loss={float(val):.6f} grads finite: {gfin}")
        failed |= not gfin
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            jax.block_until_ready(vag(*ops))
            times.append(time.perf_counter() - t0)
        print(f"[{args.mode}] grad step {min(times) * 1e3:.1f}ms (best of {args.iters})")

    if args.mode == "tiny":
        out_r, st_r = jax.jit(fwd_recurrent)(*ops)
        d_out, d_st = maxdiff(out, out_r), maxdiff(st, st_r)
        tol = 1e-2 if dtype == jnp.float32 else 6e-2
        print(f"[tiny] vs recurrent: max|out-ref|={d_out:.3e} max|state-ref|={d_st:.3e} tol={tol:.0e}")
        failed |= not (d_out < tol and d_st < tol)

    stats = jax.local_devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    if peak is not None:
        print(f"[{args.mode}] peak HBM device0: {peak / 2**30:.2f} GiB")
    print(f"[{args.mode}]", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
