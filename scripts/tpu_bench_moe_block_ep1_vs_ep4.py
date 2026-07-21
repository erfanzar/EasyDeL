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

"""MEASURED attribution: real-config qwen3_5_moe MoE block fwd+bwd,
ep1 weight-gather (the 64s regime) vs ep4 carries-batch (the 1010s regime).

The ep1-vs-ep4 step-time differential (~946s at 131k) is structurally ONLY the
MoE block: attention, GDN, dense layers and the teacher forward process the
same tokens/core on both meshes (256 batch ways either way). So timing ONE MoE
block fwd+bwd on both meshes, at the production 131k per-core volume and the
real 35B-A3B MoE dims (256 experts, top-8, H=2048, moe M=512), and multiplying
by 40 layers, reconstructs the differential and localises it.

For the ep4 arm we also dump the optimized HLO (single block -> small, no xprof,
no 2GB overflow) and print the op-category breakdown (bytes-accessed + FLOPs)
so the dominant cost category is visible directly.

MEASUREMENT RUN ONLY: random weights, a handful of timed steps, no checkpoints.
One job at a time on the slice.

    /home/erfan/easy-venv/bin/python scripts/tpu_bench_moe_block_ep1_vs_ep4.py
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

HLO_DIR = "/tmp/hlo_moe_block"


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
                "XLA_FLAGS": f"--xla_dump_to={HLO_DIR} --xla_dump_hlo_as_text",
                "REPRO_SHAPE": os.environ.get("REPRO_SHAPE", "131k"),
                "REPRO_TIME_STEPS": os.environ.get("REPRO_TIME_STEPS", "3"),
            },
        ),
    )
)
@ray.remote
def main():
    import os
    import shutil
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
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from easydel.infra.base_module import _parameter_init_sharding_context
    from easydel.modules.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
    from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig

    AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
    # Real qwen3.6-35B-A3B MoE dims.
    H = 2048
    EXPERTS = 256
    K = 8
    MOE_M = 512
    PDTYPE = jnp.bfloat16
    TIME_STEPS = int(os.environ.get("REPRO_TIME_STEPS", "3"))
    CHUNK = 65536

    shape = os.environ.get("REPRO_SHAPE", "131k")
    if shape == "131k":
        S, B = 131072, 256
    else:
        S, B = 8192, 512  # 16384 tokens/core at 256 ways

    CAND_DIMS = (1, 1, 64, 4, 4, 1)  # ep4 carries-batch (the 1010s regime)
    REF_DIMS = (1, 4, 64, 1, 4, 1)  # ep1 weight-gather (the 64s regime)
    SEED = 123

    def build(dims, *, ep_carries_batch):
        config = Qwen3NextConfig(
            hidden_size=H,
            intermediate_size=H,
            moe_intermediate_size=MOE_M,
            shared_expert_intermediate_size=MOE_M,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            num_experts=EXPERTS,
            num_experts_per_tok=K,
            norm_topk_prob=True,
            vocab_size=128,
            sharding_axis_dims=dims,
            sharding_axis_names=AXIS_NAMES,
            scan_layers=False,
        )
        config.add_basic_configurations(
            use_ring_of_experts=False,
            fsdp_is_ep_bound=False,
            sp_is_ep_bound=False,
            moe_chunk_size=CHUNK,
            moe_fsdp_shard_expert_weights=True,
            moe_ep_carries_batch=ep_carries_batch,
        )
        with _parameter_init_sharding_context(config):
            return Qwen3NextSparseMoeBlock(config=config, dtype=PDTYPE, param_dtype=PDTYPE, rngs=spx.Rngs(SEED))

    def make_x(mesh, batch_axes):
        sh = NamedSharding(mesh.jax_mesh, P(batch_axes, None, None))
        return jax.jit(lambda k: jax.random.normal(k, (B, S, H), PDTYPE) * 0.1, out_shardings=sh)(jax.random.PRNGKey(0))

    def loss_fn(m, xin):
        o = m(xin)
        o = o[0] if isinstance(o, tuple) else o
        return jnp.mean(o.astype(jnp.float32) ** 2)

    step = spx.jit(lambda m, xin: spx.value_and_grad(loss_fn)(m, xin))

    def run(tag, block, batch_axes):
        with block.config.mesh:
            x = make_x(block.config.mesh, batch_axes)
            t0 = time.perf_counter()
            loss, grads = step(block, x)
            jax.block_until_ready((loss, grads))
            compile_s = time.perf_counter() - t0
            leaves = jax.tree_util.tree_leaves(grads)
            gfin = all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
            times = []
            for _ in range(TIME_STEPS):
                t0 = time.perf_counter()
                loss, grads = step(block, x)
                jax.block_until_ready((loss, grads))
                times.append(time.perf_counter() - t0)
            mean_s = sum(times) / max(1, len(times))
            best_s = min(times)
            print(
                f"[{tag}] loss={float(loss):.6e} grad_fin={gfin} "
                f"compile+1st={compile_s:.1f}s fwdbwd_mean={mean_s * 1e3:.0f}ms "
                f"fwdbwd_best={best_s * 1e3:.0f}ms  (x40 layers -> {mean_s * 40:.1f}s)",
                flush=True,
            )
            return mean_s

    print(
        f"[bench] devices={jax.device_count()} shape={shape} B={B} S={S} H={H} E={EXPERTS} K={K} "
        f"moe_M={MOE_M} tokens/core@256ways={(B * S) // 256} chunk={CHUNK} steps={TIME_STEPS}",
        flush=True,
    )

    ref_s = cand_s = None
    try:
        ref_s = run("REF  ep1  weightgather", build(REF_DIMS, ep_carries_batch=False), ("dp", "fsdp"))
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"[REF] EXC {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    # Isolate the ep4 (CAND) module's HLO: clear the dump dir right before it compiles.
    if jax.process_index() == 0:
        try:
            shutil.rmtree(HLO_DIR, ignore_errors=True)
            os.makedirs(HLO_DIR, exist_ok=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[hlo] could not clear dump dir: {_e}", flush=True)

    try:
        cand_s = run("CAND ep4  carriesbatch", build(CAND_DIMS, ep_carries_batch=True), ("dp", "fsdp", "ep"))
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"[CAND] EXC {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    if ref_s is not None and cand_s is not None:
        print(
            f"[diff] per-layer fwd+bwd: ref(ep1)={ref_s * 1e3:.0f}ms cand(ep4)={cand_s * 1e3:.0f}ms "
            f"ratio={cand_s / max(ref_s, 1e-9):.1f}x  extrapolated x40 layers delta="
            f"{(cand_s - ref_s) * 40:.1f}s (student fwd+bwd only)",
            flush=True,
        )

    # ---- HLO op-category attribution of the ep4 block step ----
    if jax.process_index() == 0:
        _parse_hlo(HLO_DIR)

    print("[bench] done", flush=True)
    return "ok"


def _parse_hlo(hlo_dir: str) -> None:
    """Categorise the biggest optimized-HLO module by bytes-accessed and FLOPs."""
    import glob
    import re
    from collections import defaultdict

    files = glob.glob(os.path.join(hlo_dir, "*after_optimizations.txt"))
    if not files:
        files = glob.glob(os.path.join(hlo_dir, "*.txt"))
    if not files:
        print(f"[hlo] no HLO text under {hlo_dir}", flush=True)
        return
    path = max(files, key=os.path.getsize)
    print(f"[hlo] parsing {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)

    dt_bytes = {"pred": 1, "s8": 1, "u8": 1, "bf16": 2, "f16": 2, "s16": 2, "u16": 2,
                "f32": 4, "s32": 4, "u32": 4, "f64": 8, "s64": 8, "u64": 8}
    shape_re = re.compile(r"\b(pred|[suf]\d{1,2})\[([\d,]*)\]")
    inst_re = re.compile(r"^\s*%?[\w\.\-]+\s*=\s*.*?\b([a-z][a-z0-9\-]*)\(")

    def elems(dims: str) -> int:
        if dims == "":
            return 1
        n = 1
        for d in dims.split(","):
            if d.strip().isdigit():
                n *= int(d.strip())
        return n

    def line_bytes(line: str) -> int:
        total = 0
        for dt, dims in shape_re.findall(line):
            total += elems(dims) * dt_bytes.get(dt, 4)
        return total

    cat_bytes: dict[str, float] = defaultdict(float)
    cat_count: dict[str, int] = defaultdict(int)
    biggest: list[tuple[float, str]] = []
    with open(path) as f:
        for line in f:
            m = inst_re.match(line)
            if not m:
                continue
            op = m.group(1)
            if op in ("parameter", "constant", "get-tuple-element", "tuple", "after-all", "token"):
                continue
            cat = op
            if op in ("all-to-all", "ragged-all-to-all"):
                cat = "a2a"
            elif op in ("all-gather", "all-gather-start", "all-gather-done"):
                cat = "all-gather"
            elif op in ("reduce-scatter", "all-reduce", "all-reduce-start", "all-reduce-done"):
                cat = "reduce/all-reduce"
            elif op in ("collective-permute", "collective-permute-start", "collective-permute-done"):
                cat = "collective-permute"
            elif op in ("dot", "convolution"):
                cat = "dot/conv(matmul)"
            elif op == "sort":
                cat = "sort"
            elif op == "fusion":
                cat = "fusion"
            elif op in ("copy", "copy-start", "copy-done", "transpose", "bitcast", "reshape"):
                cat = "copy/transpose/reshape"
            elif op in ("dynamic-slice", "dynamic-update-slice", "gather", "scatter"):
                cat = "gather/scatter/dyn-slice"
            b = line_bytes(line)
            cat_bytes[cat] += b
            cat_count[cat] += 1
            biggest.append((b, f"{op}: {line.strip()[:150]}"))
    total_b = sum(cat_bytes.values()) or 1.0
    print("[hlo] ===== bytes-accessed by op category (proxy for HBM traffic) =====", flush=True)
    for cat, b in sorted(cat_bytes.items(), key=lambda kv: -kv[1]):
        print(f"[hlo]   {b / 1e9:10.2f} GB  {100 * b / total_b:6.2f}%  n={cat_count[cat]:<6d} {cat}", flush=True)
    print(f"[hlo]   TOTAL bytes-accessed (single fwd+bwd, per core) = {total_b / 1e9:.2f} GB", flush=True)
    print("[hlo] ===== top-25 single instructions by bytes =====", flush=True)
    for b, desc in sorted(biggest, key=lambda kv: -kv[0])[:25]:
        print(f"[hlo]   {b / 1e9:8.2f} GB  {desc}", flush=True)


if __name__ == "__main__":
    result = main()
    if hasattr(result, "error") and result.error is not None:
        try:
            print_remote_raise(result.error)
        except Exception:  # noqa: BLE001
            print(f"Raw error: {result.error}")
            raise result.error from None
