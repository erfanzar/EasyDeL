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

"""TPU parity + timing probe for MaxText-style "ep carries batch" MoE dispatch.

``moe_ep_carries_batch=True`` shards the token batch over the expert axis and
dispatches tokens to their expert-owning shard through a batch-sharded
``ragged_all_to_all`` (DISPATCH) followed by a COMBINE back (the transposed
traffic matrix), instead of replicating the batch over ep and computing at
home. Neither collective executes on XLA:CPU, so the CPU suite only proves
the routing/offset logic (numpy-emulated) and the fwd/bwd traces; THIS probe
is the runtime proof. It runs a ``Qwen3NextSparseMoeBlock`` forward+backward
on the real slice and checks, against the ep=1 reference:

  * finite forward, finite loss, finite grads on the candidate ep=4 mesh
    with the knob ON (the ep-carries-batch dispatch path);
  * the candidate loss matches the ep=1 reference loss (EP is a pure
    sharding axis — same tokens, same experts, same math);
  * per-step wall time of knob-ON vs knob-OFF on the same ep=4 mesh and vs
    the ep=1 reference, for an 8k-shaped and (optionally) 131k-shaped
    config — the a2a-cost readout the layout planner predicts at
    537 MB/dir/layer (8k) and 4.3 GB/dir/layer (131k).

MEASUREMENT RUN ONLY: a handful of steps, no checkpoints, kill after it
prints. Do not launch while a production job holds the slice.

Run from the repo root on the eray-connected dev VM (one job at a time on
the slice):

    /home/erfan/easy-venv/bin/eray run -- \
        /home/erfan/easy-venv/bin/python scripts/tpu_probe_ep_carries_batch.py
    /home/erfan/easy-venv/bin/eray logs <jobname> -f
    /home/erfan/easy-venv/bin/eray stop <jobname>

Env knobs:
    REPRO_SHAPE           : "8k" (default: S=8192) or "131k" (S=131072) bucket
    REPRO_TOKENS_PER_CORE : per-core token target at dp*fsdp*ep=256 batch ways
        (default 16384 for 8k — the production 8k bucket; 131k forces 131072)
    REPRO_HIDDEN          : hidden/intermediate size (default 2048 to make the
        a2a volume production-shaped; use 512 for a quick smoke)
    REPRO_FORCE_XLA       : 1 forces the XLA grouped matmul (default 0: Pallas)
    REPRO_TIME_STEPS      : timed steps after the compile step (default 4)
    REPRO_RUN_KNOB_OFF    : 1 (default) also times the knob-OFF ep=4 baseline
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
                "REPRO_SHAPE": os.environ.get("REPRO_SHAPE", "8k"),
                "REPRO_TOKENS_PER_CORE": os.environ.get("REPRO_TOKENS_PER_CORE", "16384"),
                "REPRO_HIDDEN": os.environ.get("REPRO_HIDDEN", "2048"),
                "REPRO_FORCE_XLA": os.environ.get("REPRO_FORCE_XLA", "0"),
                "REPRO_TIME_STEPS": os.environ.get("REPRO_TIME_STEPS", "4"),
                "REPRO_RUN_KNOB_OFF": os.environ.get("REPRO_RUN_KNOB_OFF", "1"),
            },
        ),
    )
)
@ray.remote
def main():
    """Run ep=1 reference, ep=4 knob-off baseline, and ep=4 knob-on candidate."""
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

    import easydel as ed  # noqa: F401
    import jax
    import jax.numpy as jnp
    import spectrax as spx
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from easydel.infra.base_module import _parameter_init_sharding_context
    from easydel.modules.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
    from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig

    AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
    H = int(os.environ.get("REPRO_HIDDEN", "2048"))
    EXPERTS = 64
    K = 8
    PDTYPE = jnp.bfloat16
    FORCE_XLA = os.environ.get("REPRO_FORCE_XLA", "0") == "1"
    TIME_STEPS = int(os.environ.get("REPRO_TIME_STEPS", "4"))
    RUN_KNOB_OFF = os.environ.get("REPRO_RUN_KNOB_OFF", "1") == "1"

    shape = os.environ.get("REPRO_SHAPE", "8k")
    if shape == "131k":
        S = 131072
        B = 256  # 131,072 tokens/core at 256 batch ways — the production bucket
    else:
        S = 8192
        tpc = int(os.environ.get("REPRO_TOKENS_PER_CORE", "16384"))
        # 256 batch ways (dp*fsdp on ref / dp*fsdp*ep on candidate); keep B a
        # multiple of 256 so every mesh divides it.
        B = max(256, ((tpc * 256) // S // 256) * 256)
        B = ((B + 255) // 256) * 256
    CHUNK = 65536
    CAND_DIMS = (1, 4, 16, 4, 4, 1)  # dp*fsdp*ep = 256 batch ways with the knob
    REF_DIMS = (1, 4, 64, 1, 4, 1)  # dp*fsdp = 256 batch ways, ep=1
    SEED = 123

    def build(dims, *, ep_carries_batch, chunk=CHUNK):
        """Build a Qwen3NextSparseMoeBlock under the parameter-init context."""
        config = Qwen3NextConfig(
            hidden_size=H,
            intermediate_size=H,
            moe_intermediate_size=H,
            shared_expert_intermediate_size=H,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_experts=EXPERTS,
            num_experts_per_tok=K,
            norm_topk_prob=True,
            vocab_size=128,
            moe_force_xla_gmm=FORCE_XLA,
            sharding_axis_dims=dims,
            sharding_axis_names=AXIS_NAMES,
            scan_layers=False,
        )
        config.add_basic_configurations(
            use_ring_of_experts=False,
            fsdp_is_ep_bound=False,
            sp_is_ep_bound=False,
            moe_chunk_size=chunk,
            moe_fsdp_shard_expert_weights=True,
            moe_ep_carries_batch=ep_carries_batch,
        )
        with _parameter_init_sharding_context(config):
            return Qwen3NextSparseMoeBlock(config=config, dtype=PDTYPE, param_dtype=PDTYPE, rngs=spx.Rngs(SEED))

    def make_x(mesh, batch_axes):
        """Deterministic (B, S, H) batch sharded over the given batch axes."""
        sh = NamedSharding(mesh.jax_mesh, P(batch_axes, None, None))
        return jax.jit(lambda k: jax.random.normal(k, (B, S, H), PDTYPE) * 0.1, out_shardings=sh)(jax.random.PRNGKey(0))

    def loss_fn(m, xin):
        o, _ = m(xin)
        return jnp.mean(o.astype(jnp.float32) ** 2), o

    step = spx.jit(lambda m, xin: spx.value_and_grad(loss_fn, has_aux=True)(m, xin))

    def run(tag, block, batch_axes):
        """Compile + parity + steady-state step timing; prints one report line."""
        with block.config.mesh:
            x = make_x(block.config.mesh, batch_axes)

            t0 = time.perf_counter()
            (loss, out), grads = step(block, x)
            jax.block_until_ready((loss, out))
            compile_s = time.perf_counter() - t0

            leaves = jax.tree_util.tree_leaves(grads)
            gfin = all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
            gmax = max(float(jnp.max(jnp.abs(jnp.where(jnp.isfinite(g), g, 0.0)))) for g in leaves)
            ofin = bool(jnp.all(jnp.isfinite(out)))
            omax = float(jnp.max(jnp.abs(jnp.where(jnp.isfinite(out), out, 0.0))))

            times = []
            for _ in range(TIME_STEPS):
                t0 = time.perf_counter()
                (loss, out), grads = step(block, x)
                jax.block_until_ready((loss, out, grads))
                times.append(time.perf_counter() - t0)
            mean_s = sum(times) / max(1, len(times))
            best_s = min(times) if times else float("nan")

            print(
                f"[{tag}] loss={float(loss):.6e} loss_fin={bool(jnp.isfinite(loss))} "
                f"fwd_fin={ofin} fwd_max={omax:.3e} grad_fin={gfin} grad_max={gmax:.3e} "
                f"compile+step1={compile_s:.2f}s step_mean={mean_s * 1e3:.1f}ms step_best={best_s * 1e3:.1f}ms",
                flush=True,
            )
            return float(loss)

    print(
        f"[probe] devices={jax.device_count()} shape={shape} B={B} S={S} H={H} K={K} experts={EXPERTS} "
        f"tokens/core@256ways={(B * S) // 256} chunk={CHUNK} force_xla={FORCE_XLA} steps={TIME_STEPS}",
        flush=True,
    )

    ref_loss = None
    try:
        ref_loss = run("REF  ep1        fsdpW", build(REF_DIMS, ep_carries_batch=False), ("dp", "fsdp"))
    except Exception as e:  # noqa: BLE001
        print(f"[REF  ep1] EXC {type(e).__name__}: {e}", flush=True)
    if RUN_KNOB_OFF:
        try:
            run("BASE ep4 knobOFF fsdpW", build(CAND_DIMS, ep_carries_batch=False), ("dp", "fsdp"))
        except Exception as e:  # noqa: BLE001
            print(f"[BASE ep4 knobOFF] EXC {type(e).__name__}: {e}", flush=True)
    cand_loss = None
    try:
        cand_loss = run("CAND ep4 knobON  fsdpW", build(CAND_DIMS, ep_carries_batch=True), ("dp", "fsdp", "ep"))
    except Exception as e:  # noqa: BLE001
        print(f"[CAND ep4 knobON] EXC {type(e).__name__}: {e}", flush=True)

    if ref_loss is not None and cand_loss is not None:
        rel = abs(cand_loss - ref_loss) / max(abs(ref_loss), 1e-30)
        verdict = "PASS" if rel < 5e-3 else "FAIL"
        print(f"[parity] ref={ref_loss:.6e} cand={cand_loss:.6e} rel_diff={rel:.3e} -> {verdict}", flush=True)

    print("[probe] done", flush=True)
    return "ok"


if __name__ == "__main__":
    result = main()
    if hasattr(result, "error") and result.error is not None:
        try:
            print_remote_raise(result.error)
        except Exception:  # noqa: BLE001
            print(f"Raw error: {result.error}")
            raise result.error from None
