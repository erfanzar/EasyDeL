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
    REPRO_HIDDEN          : hidden size H (default 2048 so the a2a volume is
        production-shaped; the MoE intermediate is fixed at 512 like
        production so gmm shapes match the real regime)
    REPRO_FORCE_XLA       : 1 forces the XLA grouped matmul (default 0: Pallas)
    REPRO_TIME_STEPS      : timed steps after the compile step (default 4)
    REPRO_RUN_KNOB_OFF    : 1 (default) also times the knob-OFF ep=4 baseline

Note on tiling (first-run failure, job 20260717-073048): the config default
``moe_tiling_size_batch=4`` fails the ``% 8`` check in ``_sparse_moe_call``
and silently falls back to the XLA ``ragged_dot`` with ``bypass_xla_tiling``;
XLA's ragged_dot_expander then materializes a gathered per-8-row-tile RHS of
``[m/8, K, N]`` — ``bf16[16384, 2048, 4096]`` = 2^32 64-byte words at the
first probe's shapes, tripping the jellyfish int32 bounds check in EVERY arm
(ep=1 included). The probe pins ``moe_tiling_size_batch/dim = 1024`` so the
fused path takes the real Pallas grouped matmul, as production does.
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
            moe_intermediate_size=512,  # production M; keeps gmm shapes in the real Pallas regime
            shared_expert_intermediate_size=512,
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
            # Pallas-valid tiles: the default moe_tiling_size_batch=4 fails the
            # %8 check and drops into the XLA ragged_dot expander, whose
            # gathered [m/8, K, N] RHS trips the int32 bounds check here.
            moe_tiling_size_batch=1024,
            moe_tiling_size_dim=1024,
        )
        with _parameter_init_sharding_context(config):
            return Qwen3NextSparseMoeBlock(config=config, dtype=PDTYPE, param_dtype=PDTYPE, rngs=spx.Rngs(SEED))

    def loss_fn(m, xin):
        o, _ = m(xin)
        return jnp.mean(o.astype(jnp.float32) ** 2), o

    def run(tag, block, batch_axes):
        """Compile + parity + steady-state step timing; prints one report line.

        Each arm is ONE jitted program returning scalars only. The first two
        probe runs died on cross-program traffic: (a) the RNG-based global
        batch generator can replicate its f32 intermediate per device when
        the threefry lowering is not partitioned, and (b) returning the full
        ``[B, S, H]`` output (plus grads) out of the step and reducing it
        eagerly holds tens of GB of arrays and loaded-program reservations
        across differently-ordered meshes, which JAX bridges with an
        ``_identity_fn`` reshard program that could not reserve its arena
        (``RuntimeProgramAllocationFailure``, 64 G needed / 31 G free).
        Generating the batch from broadcasted iota math INSIDE the step
        (trivially partitionable, deterministic, identical across meshes so
        the parity check stays valid) and reducing finiteness/max stats
        in-graph leaves only scalars to cross the program boundary. A fresh
        ``spx.jit`` per arm avoids reusing another mesh's trace.
        """

        def gen_x():
            """Deterministic pseudo-random (B, S, H) batch from iota math."""
            b_i = jax.lax.broadcasted_iota(jnp.uint32, (B, S, H), 0)
            s_i = jax.lax.broadcasted_iota(jnp.uint32, (B, S, H), 1)
            h_i = jax.lax.broadcasted_iota(jnp.uint32, (B, S, H), 2)
            v = b_i * jnp.uint32(2654435761) + s_i * jnp.uint32(40503) + h_i * jnp.uint32(2246822519)
            v = v ^ (v >> 15)
            x = (v & jnp.uint32(0xFFFF)).astype(jnp.float32) / 65536.0  # [0, 1)
            x = ((x - 0.5) * 0.2).astype(PDTYPE)
            sh = NamedSharding(block.config.mesh.jax_mesh, P(batch_axes, None, None))
            return jax.lax.with_sharding_constraint(x, sh)

        def step_fn(m):
            """One measured step: gen + fwd + bwd + in-graph reductions -> scalars."""
            (loss, o), grads = spx.value_and_grad(loss_fn, has_aux=True)(m, gen_x())
            ofin = jnp.all(jnp.isfinite(o))
            omax = jnp.max(jnp.abs(jnp.where(jnp.isfinite(o), o, 0)).astype(jnp.float32))
            gfin = jnp.array(True)
            gmax = jnp.array(0.0, dtype=jnp.float32)
            for g in jax.tree_util.tree_leaves(grads):
                gfin = jnp.logical_and(gfin, jnp.all(jnp.isfinite(g)))
                gmax = jnp.maximum(gmax, jnp.max(jnp.abs(jnp.where(jnp.isfinite(g), g, 0)).astype(jnp.float32)))
            return loss, ofin, omax, gfin, gmax

        step = spx.jit(step_fn)
        with block.config.mesh:
            t0 = time.perf_counter()
            stats = step(block)
            jax.block_until_ready(stats)
            compile_s = time.perf_counter() - t0

            times = []
            for _ in range(TIME_STEPS):
                t0 = time.perf_counter()
                stats = step(block)
                jax.block_until_ready(stats)
                times.append(time.perf_counter() - t0)
            mean_s = sum(times) / max(1, len(times))
            best_s = min(times) if times else float("nan")

            loss, ofin, omax, gfin, gmax = (float(s) for s in stats)
            print(
                f"[{tag}] loss={loss:.6e} loss_fin={loss == loss} "
                f"fwd_fin={bool(ofin)} fwd_max={omax:.3e} grad_fin={bool(gfin)} grad_max={gmax:.3e} "
                f"compile+step1={compile_s:.2f}s step_mean={mean_s * 1e3:.1f}ms step_best={best_s * 1e3:.1f}ms",
                flush=True,
            )
            return loss

    print(
        f"[probe] devices={jax.device_count()} shape={shape} B={B} S={S} H={H} K={K} experts={EXPERTS} "
        f"tokens/core@256ways={(B * S) // 256} chunk={CHUNK} force_xla={FORCE_XLA} steps={TIME_STEPS}",
        flush=True,
    )

    def arm(tag, dims, *, ep_carries_batch, batch_axes):
        """Build, run, and fully release one arm (buffers + loaded programs)."""
        import gc

        try:
            block = build(dims, ep_carries_batch=ep_carries_batch)
            loss = run(tag, block, batch_axes)
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] EXC {type(e).__name__}: {e}", flush=True)
            loss = None
        finally:
            # Drop parameter buffers and unload compiled programs before the
            # next arm compiles on a different mesh: loaded-program arena
            # reservations otherwise accumulate across arms.
            block = None  # noqa: F841
            gc.collect()
            jax.clear_caches()
        return loss

    ref_loss = arm("REF  ep1        fsdpW", REF_DIMS, ep_carries_batch=False, batch_axes=("dp", "fsdp"))
    if RUN_KNOB_OFF:
        arm("BASE ep4 knobOFF fsdpW", CAND_DIMS, ep_carries_batch=False, batch_axes=("dp", "fsdp"))
    cand_loss = arm("CAND ep4 knobON  fsdpW", CAND_DIMS, ep_carries_batch=True, batch_axes=("dp", "fsdp", "ep"))

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
