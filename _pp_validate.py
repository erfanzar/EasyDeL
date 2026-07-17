"""Offline CPU validation for the PP distillation design (Phase A).

Run under the CPU trio:
  ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
  XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  nice -n 10 /home/erfan/easy-venv/bin/python _pp_validate.py
"""

import os
import sys

for _lib in ("eray", "easydel", "ejkernel", "eformer", "spectrax"):
    _p = os.path.abspath(os.path.join("libs", _lib))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import jax

print(f"jax devices: {jax.device_count()}  platform={jax.default_backend()}")

# ---------------------------------------------------------------------------
# 1. MPMD mesh construction on the fake 8-device CPU host (scaled-down PP).
# ---------------------------------------------------------------------------
from easydel.infra.base_config import EasyDeLBaseConfig

NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
scaled = (2, 1, 2, 1, 1, 2)  # pp2 * fsdp2 * sp2 = 8 -> mirrors the real (pp,fsdp,sp) structure
mesh = EasyDeLBaseConfig.create_mesh(sharding_axis_dims=scaled, sharding_axis_names=NAMES)
print("\n[1] MPMD mesh (scaled pp2/fsdp2/sp2 = 8 CPU devices)")
print(f"    shape          = {dict(mesh.shape)}")
print(f"    is_mpmd        = {getattr(mesh, 'is_mpmd', None)}")
print(f"    mpmd_axis      = {getattr(mesh, 'mpmd_axis', None)}")
print(f"    mpmd_mesh.mpmd_dim = {mesh.mpmd_mesh.mpmd_dim}")
assert mesh.is_mpmd, "scaled pp2 mesh must be MPMD"
assert int(mesh.mpmd_mesh.mpmd_dim) == 2, "mpmd_dim must equal pp=2"
# per-stage SPMD submesh should be fsdp2*sp2 = 4 devices
sub = mesh.mpmd_mesh.submesh(0)
print(f"    stage-0 submesh shape = {dict(sub.shape)}  ({sub.devices.size} devices)")
assert sub.devices.size == 4, "each stage submesh must be 8/pp = 4 devices"

# pp=1 must stay SPMD (sanity: the guard is pp>1)
spmd = EasyDeLBaseConfig.create_mesh(sharding_axis_dims=(1, 1, 8, 1, 1, 1), sharding_axis_names=NAMES)
print(f"    control: pp=1 mesh is_mpmd = {getattr(spmd, 'is_mpmd', None)} (expect False)")
assert not spmd.is_mpmd, "pp=1 must not be MPMD"

# ---------------------------------------------------------------------------
# 2. Std1F1B schedule: peak activations + microbatch/stage constraints.
# ---------------------------------------------------------------------------
from spectrax.runtime import GPipe, Std1F1B, ZeroBubbleH1

print("\n[2] Schedules")
for PP in (4, 2):
    s = Std1F1B(microbatches=16)
    peak = s.peak_activations(PP)
    print(f"    Std1F1B(N=16) pp={PP}: peak_activations/stage = {peak}  (== pp: {peak == PP})")
    assert peak == PP, "Std1F1B peak stashes must equal n_stages (M-independent)"
gp = GPipe(microbatches=16).peak_activations(4)
zb = ZeroBubbleH1(microbatches=16).peak_activations(4)
print(f"    GPipe(N=16) pp=4 peak = {gp} (all M; memory-heavy)   ZeroBubbleH1(N=16) pp=4 peak = {zb}")

# ---------------------------------------------------------------------------
# 3. Microbatch divisibility (the hard MPMD asserts): B/N and (B/N) % (fsdp*dp).
# ---------------------------------------------------------------------------
print("\n[3] Microbatch divisibility for the chosen mesh (fsdp*dp = 16), N=16")


def check(bucket, B, N, batch_ways, pp):
    ok_mod = (B % N) == 0
    per_mb = B // N if ok_mod else None
    ok_shard = ok_mod and (per_mb % batch_ways == 0)
    bubble = (pp - 1) / (N + pp - 1)
    print(
        f"    {bucket:6s} B={B:4d} N={N:2d} -> B/N={per_mb} ; B%N==0:{ok_mod} ; "
        f"(B/N)%%(fsdp*dp={batch_ways})==0:{ok_shard} ; bubble={bubble:.3f}"
    )
    return ok_shard


assert check("8k", 512, 16, 16, 4)
assert check("131k", 256, 16, 16, 4)
# show why the user's proposed (fsdp*dp=256) mesh CANNOT microbatch:
print("    --- user's proposed mesh (pp4,dp4,fsdp64) => fsdp*dp=256 ---")
check("8k", 512, 16, 256, 4)   # fails
check("131k", 256, 16, 256, 4)  # fails
check("131k", 256, 1, 256, 4)   # only N=1 works => 75% bubble

# ---------------------------------------------------------------------------
# 4. MoE layout planner on the REAL v5p per-stage submesh (pure shape math).
#    qwen3.6-35B-A3B: H=2048, E=256, moe_intermediate=512, top_k=8, 40 layers.
#    One layer's expert params (gate+up+down, all 256 experts), bf16:
# ---------------------------------------------------------------------------
from easydel.layers.moe._layout_planner import estimate_moe_layout

H = 2048
E = 256
MOE_INTER = 512
TOPK = 8
expert_params_bytes = E * (H * MOE_INTER + H * MOE_INTER + MOE_INTER * H) * 2  # bf16
print(f"\n[4] MoE layout planner (per stage-submesh; expert params/layer = {expert_params_bytes/1e9:.2f} GB bf16)")

# per-stage submesh for the recommended mesh: dp1, fsdp16, ep1, tp1, sp16
stage_axes = {"dp": 1, "fsdp": 16, "ep": 1, "tp": 1, "sp": 16}
for bucket, B, S in (("8k", 512, 8192), ("131k", 256, 131072)):
    # tokens_per_device = global_tokens / total_devices_of_listed_axes
    total_dev = 1 * 16 * 1 * 1 * 16  # per stage
    global_tokens = B * S
    tokens_per_device = global_tokens // total_dev
    est = estimate_moe_layout(
        stage_axes,
        tokens_per_device=tokens_per_device,
        num_experts_per_tok=TOPK,
        hidden_size=H,
        num_experts=E,
        expert_params_bytes=expert_params_bytes,
        dtype_bytes=2,
        fsdp_is_ep_bound=False,
        use_ring_of_experts=False,
        chunk_size=65536,
        fsdp_shard_expert_weights=True,
        ep_carries_batch=False,
    )
    print(f"    [{bucket}] {est.summary}")
    print(
        f"        -> expert_resident/dev={est.expert_weights_bytes_per_device/1e6:.1f} MB "
        f"dispatch_buf/core={est.dispatch_buffer_bytes/1e9:.2f} GB (x3 live={est.live_dispatch_bytes/1e9:.2f} GB) "
        f"hard_fail={est.exceeds_int32_allocation_bound}"
    )
    assert not est.exceeds_int32_allocation_bound, f"{bucket} MoE layout cannot compile"

# ---------------------------------------------------------------------------
# 5. Boundary-activation stash estimate (Std1F1B peak=pp) at 131k.
#    hidden [B/N, S, H] bf16, sharded over batch(fsdp*dp) * seq(sp).
# ---------------------------------------------------------------------------
print("\n[5] Boundary stash memory (131k, Std1F1B peak=pp=4)")
Bmb, S, Hd = 256 // 16, 131072, 2048
batch_shards, seq_shards = 16, 16
full = Bmb * S * Hd * 2
per_dev = full / (batch_shards * seq_shards)
print(f"    one boundary hidden [{Bmb},{S},{Hd}] bf16 = {full/1e9:.2f} GB total; "
      f"/ (batch16*sp16=256) = {per_dev/1e6:.1f} MB/dev ; x pp=4 peak = {4*per_dev/1e6:.1f} MB/dev")

print("\nALL CPU VALIDATION CHECKS PASSED")
