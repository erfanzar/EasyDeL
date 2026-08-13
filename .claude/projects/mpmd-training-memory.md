# MPMD scheduled training OOM (30B MoE, pp=4, v5p-8)

## Goal / stop condition
30B pp4 driver (`/home/erfan/.claude/jobs/41f50a25/tmp/sft_profile_30b_pp4.py`, batch 32, Std1F1B m=8,
adafactor, nothing_saveable) completes 14 steps without OOM; loss@14 in ~1.45-1.58; report root cause,
diff, step time, peak HBM before/after. Fix must stay truly MPMD (per-rank programs, real schedule,
real transfers).

## Baseline
- Command: `.venv/bin/python sft_profile_30b_pp4.py` (TPU, EJKERNEL_PERSISTENT_CACHE_DIR=/dev/shm/ejkernel-cache,
  JAX_COMPILATION_CACHE_DIR=/dev/shm/jax-cache)
- Result: E0100 RESOURCE_EXHAUSTED allocating 768MB bf16[128,2048,1536] (expert gate_up grad leaf), ~2s after
  warm compile installed. hbm-at-fail: dev0-2 in_use 15.3-15.8e9 peak 71.3-71.9e9; dev3 in_use 93.7e9 peak 102.4e9
  (v5p chip = 95.75GiB = 102.8e9 bytes → dev3 hit FULL HBM; dev3 = terminal stage).
- Preflight: m=8 phys=4 logical=4, rank_units={0:13,1:12,2:11,3:9}, body_grad_gib≈14.5GiB(=15.6e9 B)/rank,
  shared_body_grad_leaves={} (empty), bwd_transfers=24.

## Memory arithmetic (from baseline log)
- Steady in_use = params+opt ≈ 15.3-15.8e9 B/rank. One body-grad tree = 15.6e9 B.
- dev0-2 peak 71.9e9 = steady + ~3.6 grad trees. dev3 peak 102.4e9 = steady + ~5.6 grad trees.
- ⇒ multiple per-microbatch grad trees simultaneously alive per device.

## Hypotheses
- H1 residual stashes ignoring remat: structurally absent — bwd jits recompute from stage INPUTS
  (`_make_bwd_jit` does jax.vjp over the whole cluster from saved invars; only inputs (~33MB/mb) are saved).
- H2 params duplicated via placed consts: steady in_use == params+opt alone ⇒ no second copy resident. Consts are
  passed as explicit args to stage jits (`_make_fwd_jit` docstring), not baked. Deprioritized.
- H3' (leading) HOST RUN-AHEAD OVER-ALLOCATION: JAX async dispatch returns at enqueue; PJRT allocates each
  executable's OUTPUT buffers at enqueue time. Each bwd/terminal unit outputs a full fresh grad tree (15.6e9 B).
  Dispatcher threads complete units at host speed (enqueue-only), so many grad-tree outputs are allocated before
  the device catches up and the pairwise donate-merge (`_accumulate_stage_local_flat_grad_batch`,
  `_accumulate_grad_tree_donate`) can free them. Terminal rank also makes an EXTRA scaled copy per mb
  (`_scale_grad` on g_invars, runtime.py ~7048) → piles up ~2x faster → dev3 dies first. Matches: batch
  insensitivity, weight-shaped failing alloc, ~2s from install to OOM, dev3 full while dev0-2 at 71.9e9.
- H4 transport buffers retained: small (33MB/cotangent × 24) — cannot reach 56e9. Deprioritized.

## Probes
- P1 memtrace (/tmp/mpmd-mem/probe_memtrace_pp4.py + temp `_memtrace` in runtime.py; log memtrace1.out): CONFIRMS H3'.
  - warm-up fwd units enqueue at 0.2-0.4ms each (host-speed dispatch); +35MB each (small activations).
  - `stage3_terminal_fwd_mb0` enqueue: dev3 in_use 15.6→31.6e9 (+1 full grad tree ALLOCATED AT ENQUEUE).
  - after mb0 accumulate (t+1.37s): dev3 in_use 62.9e9 = params + 3 trees (fresh + `_scale_grad` copy + place/merge).
  - mb1 enqueued 4ms later while mb0 still executing → fresh tree #2 (+15.6) → then scale copies → in_use hits 93.7e9
    = EXACTLY the [hbm-at-fail] dev3 value → E0100 on a 768MB gate_up grad output leaf.
  - peak=71.9e9 on ALL devices was already set BEFORE the first schedule unit ⇒ the dev0-2 "schedule peak" from the
    baseline brief is actually a load/placement-phase high-water mark, not schedule pressure. Warm compile only
    lowers/compiles (abstract avals, `_warm_compile_schedule` docstring: never executes) — not the failure path.

## Root cause
Host-side dispatch enqueues gradient-producing stage executables far ahead of device execution; PJRT allocates each
executable's outputs (a full ~15.6e9-byte per-microbatch grad tree; on the terminal rank plus a same-size
`_scale_grad` copy) at enqueue time. The pairwise donate-merge frees trees only at device pace, so 2+ units in
flight per device ⇒ +31-62e9 transient ⇒ OOM at 30B scale. Batch-insensitive (grad trees are weight-shaped) ✓.

## P2: throttle-only run STILL OOMed identically (memtrace2/3)
Added SPX_THROTTLE/SPX_ACCUM/SPX_TERMFWD prints (memtrace3.out):
- throttle DID pop 110 handles and legitimately waited ~0.3ms (terminal exec mb0 finished during the 1.31s host
  dispatch of ~110 scale+merge ops).
- gconst_bytes=0.00GB — H2 (params-as-consts) definitively dead.
- SMOKING GUN: the 110 merge handles totaled **31.15GB = 2x the 15.61GB bf16 grad tree**. `_scale_grad(x, scale)`
  with `scale = 1/jnp.asarray(m, float32)` PROMOTES bf16→f32 (`x * f32_scalar`). Terminal accumulator AND every
  per-mb scaled tree are f32 (31.2e9 B). Terminal steady live = params 15.6 + f32 accum 31.2 + bf16 fresh 15.6 +
  f32 scaled 31.2 = 93.6e9 = the exact 93.7 fail point — OOM even with a perfect W=1 throttle.
- Downstream stages were never affected: `_cast_cotangent_like` casts the outgoing cotangent back to bf16 before
  transport, so only terminal-local trees ballooned. Root cause has TWO stacked layers: (a) f32 promotion of the
  terminal grad scale (2x terminal grad memory), (b) unbounded enqueue run-ahead multiplying in-flight trees.

## Fix layer 2 (dtype + in-place scale)
- `utils/tree.py::_scale_grad`: preserve the leaf dtype (cast product back) — kills the f32 promotion everywhere
  it is used (scheduled + legacy paths + terminal-const fold).
- `grad_core.py::_scale_grad_donate`: module-level `jax.jit(..., donate_argnums=(0,))` leaf scale.
- `runtime.py::_scale_terminal_grads`: scheduled terminal paths scale grads through the donating jit (float0/None
  passthrough) — the fresh tree is scaled IN PLACE, no second tree at all.
- Expected dev3 budget now: params 15.6 + bf16 accum 15.6 + in-flight fresh 15.6 ≈ 47e9 + temps.

## P3: post-fix runs hit E0200 RuntimeUnexpectedCoreHalt (memtrace4/5, final_validation)
`jit_bwd_rank1_f70a19074ae5d93b`: "schecklt: Invalid logical z: EncodeRemoteSyncFlagAddress() no HLO mapping" —
deterministic, at the FIRST non-terminal bwd ever reached (all OOM runs died before it, so it was masked).
A/B (same code, jax 0.10.0): COLD /dev/shm/jax-cache → 14/14 steps pass (memtrace6, final_validation_cold);
WARM cache → core halt at first bwd_rank1 execution (memtrace4, memtrace5, final_validation — 3/3). So the failing
executable is valid when compiled in-process but halts when DESERIALIZED from JAX's persistent compilation cache.

### Root cause of P3 (pre-existing spectrax bug, newly reachable)
spectrax already has `_ScopedPersistentCacheJit` (pscan_compiler.py) precisely so stage executables never touch the
persistent disk cache — but two escape hatches defeated it:
1. `_warm_compile_schedule` lowers via the guarded wrapper, then calls `lowered.compile()` RAW on worker threads →
   stage executables get WRITTEN to the global cache on cold runs and REVIVED by deserialization on warm runs.
2. `_get_schedule_direct_fused_fwd_bwd_jit` (schedule_units.py) built a plain `@jax.jit` — unguarded.

### Fix layer 3 (cache guard closure)
- `pscan_compiler._stage_persistent_cache_bypass()`: contextmanager that holds the scope lock, disables the
  persistent cache + points it at a scratch dir, resets JAX's in-process cache handle, restores on exit.
- `runtime._warm_compile_schedule` now runs its whole body (`_warm_compile_schedule_impl`) under that scope —
  worker-thread `lowered.compile()` included.
- `_get_schedule_direct_fused_fwd_bwd_jit` wraps its jit in `_scope_stage_persistent_cache(...)`.
Cost: scheduled stage executables always compile fresh per process (~42s warm-compile phase for this 30B pp4).

## FINAL RESULT (all three fix layers; /tmp/mpmd-mem/final_validation_{cold,warm}.out)
- 30B pp4 b32 driver: **14/14 steps, no OOM, BOTH cold-cache and warm-cache** (warm was 3/3 fatal before layer 3).
- Loss trajectory bit-identical across cold/warm: step1 1.656 (== SPMD fsdp=4 step1), step14 **1.547** (DoD range
  1.453-1.58; SPMD step14 = 1.453; same family, small bf16-accumulation-order divergence).
- Steady step: execution_time 9.02-9.05s/step (SPMD fsdp=4 b32 reference: 11.8s/step → pp4 is 1.31x faster despite
  idle_fraction 0.4886).
- Peak HBM: before = 102.4GB (FULL chip, dev3) + E0100; after = process peaks 76.9-83.5GB (≈72GB of that is the
  load/placement watermark predating the schedule); terminal-rank schedule cycle 47.2→62.8→47.2GB per microbatch.
- Warm-compile phase under the cache bypass: 46s (7 executables, 4 threads).
- Hygiene: full spectrax CPU suite 1793 passed / 50 skipped; lint-imports 2 kept / 0 broken; ruff clean.
- Files changed (all in libs/spectrax/spectrax/runtime/mpmd/): runtime.py, grad_core.py, pscan_compiler.py,
  schedule_units.py, utils/tree.py. Temporary probe instrumentation removed.

## Earlier intermediate result (memtrace6, cold caches, layers 1+2 only)
- 30B pp4 b32 driver: **14/14 steps, no OOM**. Steady step: dispatch_wall≈8.75s, execution_time≈9.02s
  (SPMD fsdp=4 b32 reference: 11.8s/step). step-10 loss 1.742, mean_loss 1.657 (SPMD step-10: 1.656/1.621).
- HBM at done: in_use 15.5-16.1e9/dev; peaks 75.6-78.7e9 (vs pre-schedule load watermark 71.9; baseline failure
  peaked at 102.4e9 = full HBM on dev3).
- Terminal-rank schedule cycle (memtrace4): 47.2 → 62.8 → 47.2 e9 per microbatch — bounded as designed
  (params 15.6 + bf16 accum 15.6 + one in-flight fresh 15.6 + merge transient).

---

# Session 2: scheduled-loss estimator + aux metrics + preflight caching (post-OOM fixes)

## Bugs under work (in order)
1. PP loss/grad weighting diverges from SPMD (mean-of-means vs token-weighted mean).
   Measured step-10: 1.789 (DPV) / 1.742 (ZB) vs 1.664 (SPMD), same data/order.
2. ALL scheduled runs (1F1B, ZB, DPV — not just DPV) print z_loss=None / mean_accuracy=None; SPMD
   prints z_loss 0.0 / accuracy 0.672. Root cause: the scheduled adapter loss returned only the
   scalar; spectrax's scheduled terminal enforced exactly one output; `_apply_stage_local_gradients`
   built LossMetrics(loss) only.
3. "MPMD schedule preflight" recomputed + logged per step: `_dispatch_schedule_faithful` rebuilt
   units+deps every dispatch and fused_async recomputed `_schedule_preflight_stats` every step
   (log capped at 8 but the 14-step run shows 8 = "every step" to the eye).
4. Load/placement watermark ~72GB/dev before the schedule starts — probe pending.

## Root cause for bug 1
Per-mb scheduled loss = sum(token_losses)/n_valid_mb; runtime reduced with uniform 1/M
(loss AND cotangent/const-grad scale) → mean-of-means. SPMD computes sum over batch / total valid.
Fix estimator: scale mb loss and ALL its grads by w_mb/sum(w), w_mb = mb valid-token count
(computed host-side per step, per mb, BEFORE dispatch — no cross-rank comm, downstream ranks
untouched; identity: sum((w/W)·(S_mb/w)) = S/W = SPMD).

## Implementation (spectrax)
- `_make_terminal_jit(..., n_aux=)`: terminal now returns `(loss, aux, (g_consts, g_invars))`
  via value_and_grad(has_aux=True); terminal cluster may emit 1+n_aux outputs (all must be
  produced after the last sxstage_iter — validated at plan build with a clear error).
- plan keys: terminal_n_aux, fn_out_treedef, args_treedef.
- `sxvalue_and_grad(fn, argnums, *, has_aux=False, microbatch_weight_fn=None)`; weight fn has
  fn's signature, called eagerly per mb on sliced batch args; scales resolved by
  `_resolve_microbatch_loss_scales` (plan["microbatch_weight_fn"] set/popped per call);
  `sxvalue_and_grad_and_apply` mirrors both kwargs.
- fused_async + serial + gpipe fwd/bwd dispatchers: per-mb scale = w̃_mb (weighted) or 1/M
  (unchanged uniform path, bit-compatible); weighted path scales terminal g_consts per-mb via
  the donating leaf jit (in place, no extra tree — memory profile of session-1 fixes preserved)
  and skips the final 1/M const scale; loss/aux reduced via `_weighted_terms_sum` (mb-sorted).
- `_dispatch_schedule_faithful` returns (loss, grads, aux); units+deps cached on the plan
  (`_schedule_units_cache`, keyed by apply-unit request); preflight stats cached with them and
  logged only when freshly computed (bug 3).
- sxcall legacy path intentionally unchanged (still uniform, has_aux still rejected there).

## Implementation (easydel)
- `ScheduledLossAdapter(has_aux=..., make_microbatch_weight=...)`; base adapter returns
  `(loss, {"z_loss","accuracy"})` + `_make_base_scheduled_weight` (mirrors ForCausalLMLoss /
  chunked-FLCE denominator: shifted labels, ignore_index validity ∧ shifted attention_mask /
  decoder_loss_weights; gated by `_base_scheduled_weight_semantics` — uniform for reduction=
  sum/none, divide_weight_sum, constant factors, mtp_only; "valid_only" for dft).
- `_ScheduledValueAndGradCompiler` threads (loss, grads, aux); `_apply_stage_local_gradients`
  fills LossMetrics(z_loss=…, accuracy=…); aux scalars host-replicated for multi-controller.
- gacc>1: plain mean across accumulation steps (matches SPMD minibatch_call semantics).

## Tests added (CPU)
- libs/spectrax/tests/pipeline/test_schedule_weighted_loss.py (8 tests): weighted pp2 loss+grads
  == single-device global weighted estimator (1e-5) for GPipe+Std1F1B; unweighted stays
  mean-of-means (and provably differs from global on the data); aux weighted reduction; has_aux
  mismatch errors; schedule-units cache stability.
- libs/easydel/tests/trainers/test_scheduled_loss_weighting.py (4 tests): weight fn ==
  ForCausalLMLoss.weight_sum (with/without attention mask over -100 prompts); weighted
  recombination of per-mb ForCausalLMLoss == full-batch loss to 1e-6 (uniform provably differs);
  semantics gating.
- Fixed stale monkeypatch in test_training_utils_gradient_accumulation (fake sxvalue_and_grad
  now accepts the new kwargs).
- Full spectrax CPU suite: 1801 passed / 50 skipped (baseline 1793 + 8 new). qwen3/llama/stage-region
  mpmd module tests pass (96+4); easydel/tests/trainers full: 705 passed / 1 skipped (a 139-segfault
  seen once was environmental — concurrent TPU driver + CPU suite; clean on rerun); infra mpmd tests 20 pass.

## TPU validation (v5p-8, all three drivers 14/14, no OOM, no E0100/E0200)
IMPORTANT context discovered: the SPMD "reference" numbers (1.664@10 / 1.453@14) are the b32ub
config (`fsdp_is_ep_bound: False`); the plain b32 SPMD run differs from b32ub by up to 2.3e-2
at the same step (1.508 vs 1.531 @8) purely from the sharding flag — that is the honest
run-to-run fp band for this workload. ±1e-3 is not attainable even SPMD-vs-SPMD.
- 1F1B (log1 rerun, weighted): step1 1.664 == SPMD-b32ub step1 1.664 EXACT (estimator identity
  on real data at identical params); step10 1.661 (ref 1.664, Δ3e-3); step14 1.458 (ref 1.453,
  Δ5e-3); all steps within the inter-SPMD noise band. Old mean-of-means was 1.742@10 (Δ7.8e-2).
- ZB: step10 1.659, z_loss 0.0, accuracy 0.672 (SPMD 0.672), exec 8.208s (ref 8.16, +0.6%).
- DPV: step10 1.661, z_loss 0.0, accuracy 0.676, mean_accuracy 0.667 — aux now numeric (bug 2);
  exec 5.65s at step10, steady dispatch 5.5-6.4s, wall cadence ~6.3s (ref 6.88 → ~9-18% FASTER,
  DPV benefits most from unit/dep+preflight caching: m=16 x 8 virtual stages = biggest DAG).
- 1F1B steady: dispatch_wall 8.74-8.76s, wall cadence 9.0s/step (ref 9.03) — no regression.
- Preflight logged exactly once per run (bug 3): grep count == 1 in all three driver logs.
- DPV needed a relaunch once: previous driver's libtpu vfio fds not yet released ("/dev/vfio/3
  busy") when chained immediately after ZB — infra, not code.

## Bug 4: load/placement watermark — QUANTIFIED, fix deferred as scoped follow-up
Probe: /tmp/mpmd-fix2/probe_load_watermark.py (+ watermark.out): 0.25s HBM sampler + phase
markers on place_setup_tree_with_shardings / EasyDeLState.init_tx; Std1F1B, max_steps=2.
Timeline (per device, d0 shown; all four track together):
- t=83-133s streaming tensorstore load: in_use climbs 0 -> 45GB on EVERY device — leaves are
  resident on all devices (not stage-placed) while the checkpoint streams in.
- t=133.7s single bulk step: peak jumps 45 -> 72GB (fused QKV/gate-up reform + final stage
  placement materialize transformed copies while sources are still alive), then everything
  releases to in_use 15.6GB by t=174s (params properly stage-placed).
- init_tx (adafactor): NEGLIGIBLE — enter/exit at 15.6/71.9, +0.1GB; 833 cross-device-set
  leaves host-staged but all tiny (factored rows/cols + scalars). init_tx is NOT the problem.
- Training steps after: steady 15.5-16.1 in_use; schedule-phase peaks 78-80GB (terminal grad
  cycle) — consistent with session-1's validated budget; the 72GB load watermark is fully
  separate and earlier.
Root cause: from_pretrained streaming keeps ~45GB/dev of not-yet-placed leaves alive, then does
layout-fusion + placement as one bulk step (+27GB transient). Scoped follow-up (invasive,
touches HF conversion for all models — NOT done here): stream each leaf to its final
stage-local sharding at read time and run fused-group reform incrementally per group, freeing
staging copies eagerly. Expected watermark: ~max(few layers in flight) + steady, i.e. <25GB.

## Final hygiene (session 2)
ruff check+format clean on all touched files; lint-imports 2 kept / 0 broken; focused tests
8+14 pass post-format. No commits (per instruction).

## Fix layer 1 (spectrax/runtime/mpmd/runtime.py, scheduled path)
Per-rank grad-unit run-ahead throttle `_throttle_grad_unit_runahead(rank)`:
- `_accumulate_stage_local_flat_grad_batch` now returns `(saw, merge_handles)`; `_accumulate_bwd_result` stores the
  unit's merged accumulator handles (stage-local + const merges) in `rank_grad_sync[rank]`.
- Before enqueueing the NEXT grad-producing unit on the same rank (full BWD, BWD_W, terminal fwd+loss+bwd, fused
  fwd+bwd — NOT BWD_I, NOT plain fwd), block_until_ready on the previous unit's merge handles. Merge donates the
  accumulator, so completion ⇒ previous fresh tree freed. Bound: params + accum + 1 in-flight unit.
- MPMD preserved: per-rank programs/schedule/transfers untouched; only host enqueue pacing of grad units changes.
  In steady 1F1B the interleaved fwd unit keeps the device queue non-empty during the wait (no bubble); drain-phase
  bubbles = host enqueue latency (~ms) per unit.
