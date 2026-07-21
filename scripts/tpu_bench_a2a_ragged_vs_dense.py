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

"""Microbench: ``jax.lax.ragged_all_to_all`` vs dense ``jax.lax.all_to_all``.

Times the two expert-parallel token-redistribution collectives *in isolation*
on the real v5p-2048 slice, at the exact per-core volume the qwen3.6-35B-A3B
MoE dispatch moves at 131k context (and 8k), on the production ep-carries-batch
mesh ``(1,1,64,4,4,1)``. This is the decisive cheap test for the hypothesis
that the ragged collective's data-dependent dynamic sizes defeat XLA's fast
hardware collective, while a STATIC-shape dense ``all_to_all`` runs at ICI
bandwidth.

Both arms move the SAME bytes off-core per core (``(ep-1)*cap*H`` rows), so the
GB/s comparison is apples-to-apples. The ep axis (size 4) is placed on the
identical torus lines as production (built via ``spx.create_mesh`` with the
production axis dims), and the collective runs on all 256 ep-lines at once
(realistic ICI contention).

131k volume: cap = tokens/core * k / ep = 131072 * 8 / 4 = 262144 rows/dest;
send buffer = ep*cap*H*2 = 8.59 GB/core (== the ~8.6 GB/core/dir figure).
8k volume : cap = 16384 * 8 / 4 = 32768 rows/dest; 1.07 GB/core.

MEASUREMENT RUN ONLY: a handful of timed iters, no checkpoints, no model.
Kill after it prints. One job at a time on the slice.

    /home/erfan/easy-venv/bin/python scripts/tpu_bench_a2a_ragged_vs_dense.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("EFORMER_DISABLE_FORKIFY", "1")

_eray_path = os.path.abspath(os.path.join("libs", "eray"))  # noqa
if os.path.isdir(_eray_path) and _eray_path not in sys.path:  # noqa
    sys.path.insert(0, _eray_path)  # noqa

import ray  # noqa: E402
from eray import TpuAcceleratorConfig, execute, print_remote_raise  # noqa: E402
from ray.runtime_env import RuntimeEnv  # noqa: E402

RUNTIME_WORKING_DIR = os.path.abspath(".")

if os.environ.get("RAY_JOB_CONFIG_JSON_ENV_VAR"):
    ray.init(address="auto")
else:
    ray.init(address="auto", runtime_env=RuntimeEnv(working_dir=RUNTIME_WORKING_DIR))


@execute(
    TpuAcceleratorConfig(
        tpu_version="v5p-2048",
        pod_count=1,
        execution_env=RuntimeEnv(
            env_vars={
                "EFORMER_DISABLE_FORKIFY": "1",
                "EFORMER_SUBPROCESS_TIMEOUT_S": "1000000",
                "EASYDEL_TARGETED_TPU_GENERATION": "v5p",
                "PYTHONPATH": "libs/eray:libs/easydel:libs/ejkernel:libs/eformer:libs/spectrax:.",
                "JAX_COMPILATION_CACHE_DIR": "/home/erfan/jax-compile-cache",
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "10",
                "BENCH_H": os.environ.get("BENCH_H", "2048"),
                "BENCH_K": os.environ.get("BENCH_K", "8"),
                "BENCH_ITERS": os.environ.get("BENCH_ITERS", "20"),
            },
        ),
    )
)
@ray.remote
def main():
    """ep=1 reference not needed; time ragged vs dense a2a on the ep mesh."""
    import os
    import sys
    import time

    for _lib in ("eray", "easydel", "ejkernel", "eformer", "spectrax"):
        _p = os.path.abspath(os.path.join("libs", _lib))
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    _root = os.path.abspath(".")
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import jax
    import jax.numpy as jnp
    import spectrax as spx
    from jax.sharding import PartitionSpec as P

    try:
        from jax import shard_map  # jax >= 0.10 stable
    except ImportError:
        from jax.experimental.shard_map import shard_map

    from easydel.layers.moe._communication_utils import (
        build_ep_traffic_matrix,
        ep_batch_receive_buffer_rows,
        get_all_to_all_params,
    )

    H = int(os.environ.get("BENCH_H", "2048"))
    K = int(os.environ.get("BENCH_K", "8"))
    ITERS = int(os.environ.get("BENCH_ITERS", "20"))
    EP = 4
    DT = jnp.bfloat16
    ITEMSIZE = 2

    AXIS_DIMS = (1, 1, 64, 4, 4, 1)
    AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
    mesh = spx.create_mesh(axis_dims=AXIS_DIMS, axis_names=AXIS_NAMES)
    jmesh = mesh.jax_mesh
    print(f"[bench] devices={jax.device_count()} mesh={jmesh.shape}", flush=True)

    # Per-core seed sharded over the concrete cores so shard_map maps one scalar
    # to each device. dp/pp/sp are size 1; fsdp*ep*tp = 64*4*4 = 1024 cores.
    core_axes = ("fsdp", "ep", "tp")
    n_cores = 1
    for a in core_axes:
        n_cores *= jmesh.shape[a]
    seeds = jnp.arange(n_cores, dtype=jnp.int32)
    seed_spec = P(core_axes)

    def _bytes_moved_per_core(cap: int) -> float:
        # all_to_all / ragged send (ep-1) destinations' worth of cap rows off-core.
        return (EP - 1) * cap * H * ITEMSIZE

    def bench_dense(cap: int, tag: str) -> None:
        """Chain ITERS static all_to_all over the ep axis; report GB/s."""

        def body(seed):
            key = jax.random.fold_in(jax.random.PRNGKey(0), seed.reshape(()))
            x = jax.random.normal(key, (EP * cap, H), DT)
            for _ in range(ITERS):
                x = jax.lax.all_to_all(x, "ep", split_axis=0, concat_axis=0, tiled=True)
                x = x + jnp.bfloat16(1e-3)
            return jnp.sum(x.astype(jnp.float32)).reshape(1)

        fn = jax.jit(
            shard_map(body, mesh=jmesh, in_specs=(seed_spec,), out_specs=P(core_axes), check_vma=False)
        )
        with jmesh:
            t0 = time.perf_counter()
            out = fn(seeds)
            jax.block_until_ready(out)
            compile_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            out = fn(seeds)
            jax.block_until_ready(out)
            run_s = time.perf_counter() - t0
        per = run_s / ITERS
        gbps = _bytes_moved_per_core(cap) / per / 1e9
        buf_gb = EP * cap * H * ITEMSIZE / 1e9
        print(
            f"[dense {tag}] cap={cap} buf/core={buf_gb:.2f}GB per_a2a={per * 1e3:.2f}ms "
            f"moved/core/dir={_bytes_moved_per_core(cap) / 1e9:.2f}GB GB/s={gbps:.1f} "
            f"(compile+1st {compile_s:.1f}s, {ITERS} chained)",
            flush=True,
        )

    def bench_ragged(cap: int, tag: str) -> None:
        """Chain ITERS ragged_all_to_all with balanced static send/recv sizes."""
        rows = EP * cap  # local source rows (== dense EP*cap)
        # Balanced traffic: send `cap` rows to each of EP destinations.
        off = jnp.arange(EP, dtype=jnp.int32) * cap  # [0, cap, 2cap, 3cap]
        sizes = jnp.full((EP,), cap, dtype=jnp.int32)

        def body(seed):
            key = jax.random.fold_in(jax.random.PRNGKey(0), seed.reshape(()))
            x = jax.random.normal(key, (rows, H), DT)
            for _ in range(ITERS):
                buf = jnp.zeros((rows, H), DT)
                x = jax.lax.ragged_all_to_all(x, buf, off, sizes, off, sizes, axis_name="ep")
                x = x + jnp.bfloat16(1e-3)
            return jnp.sum(x.astype(jnp.float32)).reshape(1)

        fn = jax.jit(
            shard_map(body, mesh=jmesh, in_specs=(seed_spec,), out_specs=P(core_axes), check_vma=False)
        )
        with jmesh:
            t0 = time.perf_counter()
            out = fn(seeds)
            jax.block_until_ready(out)
            compile_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            out = fn(seeds)
            jax.block_until_ready(out)
            run_s = time.perf_counter() - t0
        per = run_s / ITERS
        gbps = _bytes_moved_per_core(cap) / per / 1e9
        buf_gb = rows * H * ITEMSIZE / 1e9
        print(
            f"[ragged {tag}] cap={cap} buf/core={buf_gb:.2f}GB per_a2a={per * 1e3:.2f}ms "
            f"moved/core/dir={_bytes_moved_per_core(cap) / 1e9:.2f}GB GB/s={gbps:.1f} "
            f"(compile+1st {compile_s:.1f}s, {ITERS} chained)",
            flush=True,
        )

    E = 256  # real qwen3.6-35B-A3B routed-expert count
    LE = E // EP  # local experts per shard = 64

    def bench_ragged_dynamic(tokens_per_core: int, tag: str) -> None:
        """Faithful ep-carries-batch dispatch a2a: DATA-DEPENDENT send/recv sizes
        (from an on-device random routing) into the real worst-case recv buffer.

        This is the arm that tests the hypothesis that ragged_all_to_all's
        dynamic sizes defeat XLA's fast collective. Unlike ``bench_ragged``
        (compile-time-constant sizes), the offsets/sizes here are traced values
        derived from a bincount of random top-k assignments, exactly as the real
        ``_dispatch_ep_batch`` computes them.
        """
        T = int(tokens_per_core)
        recv_rows = ep_batch_receive_buffer_rows(T, K, EP, LE)  # static worst case

        def body(seed):
            key = jax.random.fold_in(jax.random.PRNGKey(0), seed.reshape(()))
            gate = jax.random.normal(key, (T, E), jnp.float32)
            _, top = jax.lax.top_k(gate, K)  # [T, K]
            group_sizes = jnp.bincount(top.ravel(), length=E)  # [E] data-dependent
            gg = jax.lax.all_gather(group_sizes, axis_name="ep")  # [ep, E]
            traffic = build_ep_traffic_matrix(gg, EP, LE)  # [ep, ep] dynamic
            shard_id = jax.lax.axis_index("ep")
            in_off, send, out_off, recv = get_all_to_all_params(
                traffic, shard_id, EP, is_batch_sharded=True, is_dispatch=True
            )
            x = jax.random.normal(jax.random.fold_in(key, 1), (T * K, H), DT)
            buf = jnp.zeros((recv_rows, H), DT)
            y = buf
            for _ in range(ITERS):
                y = jax.lax.ragged_all_to_all(x, buf, in_off, send, out_off, recv, axis_name="ep")
                x = y[: T * K] + jnp.bfloat16(1e-3)
            return jnp.sum(y.astype(jnp.float32)).reshape(1)

        fn = jax.jit(
            shard_map(body, mesh=jmesh, in_specs=(seed_spec,), out_specs=P(core_axes), check_vma=False)
        )
        with jmesh:
            t0 = time.perf_counter()
            out = fn(seeds)
            jax.block_until_ready(out)
            compile_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            out = fn(seeds)
            jax.block_until_ready(out)
            run_s = time.perf_counter() - t0
        per = run_s / ITERS
        moved = T * K * H * ITEMSIZE * (EP - 1) / EP  # ~balanced off-core bytes
        gbps = moved / per / 1e9
        recv_gb = recv_rows * H * ITEMSIZE / 1e9
        print(
            f"[ragged_dyn {tag}] tokens/core={T} recv_buf/core={recv_gb:.2f}GB per_a2a={per * 1e3:.2f}ms "
            f"~moved/core/dir={moved / 1e9:.2f}GB GB/s={gbps:.1f} "
            f"(compile+1st {compile_s:.1f}s, {ITERS} chained)",
            flush=True,
        )

    # 131k per-core volume: 131072 tokens/core * k=8 / ep=4 = 262144 rows/dest.
    # 8k  per-core volume: 16384  tokens/core * k=8 / ep=4 = 32768  rows/dest.
    cap_131k = 131072 * K // EP
    cap_8k = 16384 * K // EP

    for tag, cap, tpc in (("131k", cap_131k, 131072), ("8k", cap_8k, 16384)):
        for name, call in (
            ("ragged_static", lambda c=cap, t=tag: bench_ragged(c, t)),
            ("ragged_dynamic", lambda tp=tpc, t=tag: bench_ragged_dynamic(tp, t)),
            ("dense", lambda c=cap, t=tag: bench_dense(c, t)),
        ):
            try:
                call()
            except Exception as e:  # noqa: BLE001
                import traceback

                print(f"[{name} {tag}] EXC {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

    print("[bench] done", flush=True)
    return "ok"


if __name__ == "__main__":
    result = main()
    if hasattr(result, "error") and result.error is not None:
        try:
            print_remote_raise(result.error)
        except Exception:  # noqa: BLE001
            print(f"Raw error: {result.error}")
            raise result.error from None
