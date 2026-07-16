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

"""TPU parity probe for the fused-MoE expert-parallel (ep>1, non-ring) dispatch.

The non-ring ep>1 branch of ``BaseMoeModule._sparse_moe_call`` (``local_permute``
+ ``jax.lax.ragged_all_to_all``) cannot execute on XLA:CPU — ragged_all_to_all
does not lower there — so its runtime numerics are TPU-only. This probe runs a
``Qwen3NextSparseMoeBlock`` (the block qwen3_5_moe / Qwen3.6-35B-A3B uses:
FUSED gate_up kernel, shared expert, TOP_K routing) at a production-like regime
on the real slice, forward + backward, and compares an ep>1 candidate mesh
against an ep=1 reference. EP is a pure sharding axis, so a correct candidate
must stay finite and match the reference loss.

Validated 2026-07-16 on prism-v5p-2048 (1024 devices), bf16, Pallas grouped
matmul, 65536 tokens/core, moe_chunk_size=65536, moe_fsdp_shard_expert_weights:
    ep=1 (1,4,64,1,4,1): loss 5.3693e-08, all grads finite
    ep=4 (1,4,16,4,4,1): loss 5.3694e-08, all grads finite
and the equivalent GptOss (separate gate/up kernels, wd_bias) block matched
its ep=1 reference to 1.2e-4 in f32. The bf16 ep>1-vs-ep=1 elementwise spread
(max ~5.8e-2 at mean |diff| ~1e-4) is identical for the trusted ring path and
is accumulation-order noise, not a defect.

Run from the repo root on the eray-connected dev VM (one job at a time on the
slice):

    /home/erfan/easy-venv/bin/eray run -- \
        /home/erfan/easy-venv/bin/python scripts/tpu_probe_moe_ep_parity.py
    /home/erfan/easy-venv/bin/eray logs <jobname> -f
    /home/erfan/easy-venv/bin/eray stop <jobname>

Env knobs:
    REPRO_TOKENS_PER_CORE : per-core token target for the candidate mesh
        (default 65536)
    REPRO_HIDDEN          : hidden/intermediate size (default 512)
    REPRO_FORCE_XLA       : 1 forces the XLA grouped matmul (default 0: Pallas)
    REPRO_GRAD            : 1 (default) forward+backward, 0 forward-only
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
                "REPRO_TOKENS_PER_CORE": os.environ.get("REPRO_TOKENS_PER_CORE", "65536"),
                "REPRO_HIDDEN": os.environ.get("REPRO_HIDDEN", "512"),
                "REPRO_FORCE_XLA": os.environ.get("REPRO_FORCE_XLA", "0"),
                "REPRO_GRAD": os.environ.get("REPRO_GRAD", "1"),
            },
        ),
    )
)
@ray.remote
def main():
    """Run the ep=1 reference and the ep=4 candidate on every pod host."""
    import os
    import sys

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
    H = int(os.environ.get("REPRO_HIDDEN", "512"))
    EXPERTS = 64
    K = 8
    PDTYPE = jnp.bfloat16
    FORCE_XLA = os.environ.get("REPRO_FORCE_XLA", "0") == "1"
    DO_GRAD = os.environ.get("REPRO_GRAD", "1") == "1"

    tpc = int(os.environ.get("REPRO_TOKENS_PER_CORE", "65536"))
    S = 8192
    B = max(256, ((tpc * 64) // S // 256) * 256)
    B = ((B + 255) // 256) * 256
    CHUNK = 65536
    CAND_DIMS = (1, 4, 16, 4, 4, 1)
    REF_DIMS = (1, 4, 64, 1, 4, 1)
    SEED = 123

    def build(dims, *, ring, chunk, fsdp_shard):
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
            use_ring_of_experts=ring,
            fsdp_is_ep_bound=False,
            sp_is_ep_bound=False,
            moe_chunk_size=chunk,
            moe_fsdp_shard_expert_weights=fsdp_shard,
        )
        with _parameter_init_sharding_context(config):
            return Qwen3NextSparseMoeBlock(config=config, dtype=PDTYPE, param_dtype=PDTYPE, rngs=spx.Rngs(SEED))

    def make_x(mesh):
        """Deterministic (B, S, H) batch sharded over (dp, fsdp)."""
        sh = NamedSharding(mesh.jax_mesh, P(("dp", "fsdp"), None, None))
        return jax.jit(lambda k: jax.random.normal(k, (B, S, H), PDTYPE) * 0.1, out_shardings=sh)(jax.random.PRNGKey(0))

    def fin(a):
        """(all-finite?, max |finite part|) of an array."""
        return bool(jnp.all(jnp.isfinite(a))), float(jnp.max(jnp.abs(jnp.where(jnp.isfinite(a), a, 0.0))))

    def run(block):
        """Forward (+optional backward); return a finiteness report string."""
        with block.config.mesh:
            x = make_x(block.config.mesh)
            if DO_GRAD:

                def loss_fn(m, xin):
                    o, _ = m(xin)
                    return jnp.mean(o.astype(jnp.float32) ** 2), o

                (loss, out), grads = spx.value_and_grad(loss_fn, has_aux=True)(block, x)
                leaves = jax.tree_util.tree_leaves(grads)
                gfin = all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
                gmax = max(float(jnp.max(jnp.abs(jnp.where(jnp.isfinite(g), g, 0.0)))) for g in leaves)
                ofin, omax = fin(out)
                return (
                    f"fwd_fin={ofin} fwd_max={omax:.3e} loss={float(loss):.4e} "
                    f"loss_fin={bool(jnp.isfinite(loss))} grad_fin={gfin} grad_max={gmax:.3e}"
                )
            out, _ = block(x)
            ofin, omax = fin(out)
            return f"fwd_fin={ofin} fwd_max={omax:.3e}"

    print(
        f"[probe] devices={jax.device_count()} B={B} S={S} H={H} K={K} experts={EXPERTS} "
        f"tpc~{(B // 64) * S} chunk={CHUNK} force_xla={FORCE_XLA} grad={DO_GRAD}",
        flush=True,
    )
    try:
        print(f"[REF  ep1 fused fsdpW ] {run(build(REF_DIMS, ring=False, chunk=CHUNK, fsdp_shard=True))}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[REF  ep1] EXC {type(e).__name__}: {e}", flush=True)
    try:
        print(f"[CAND ep4 fused fsdpW ] {run(build(CAND_DIMS, ring=False, chunk=CHUNK, fsdp_shard=True))}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[CAND ep4] EXC {type(e).__name__}: {e}", flush=True)

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
