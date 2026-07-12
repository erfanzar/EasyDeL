# verl RL-infra parity — roadmap

Bringing verl's RL-infrastructure advantages into EasyDeL (JAX/TPU-native),
grounded in the actual code. Status legend: ☐ todo · ◐ in progress · ☑ done.

## Context

EasyDeL already matches or exceeds verl on RL *algorithms/features* (RLVR
verifiers incl. code execution, LoRA-in-RL, multi-turn tool-calling via
agentic_moshpit, async-GRPO, sequence packing, PPO+value+GAE, and a broader
offline+preference+distillation set). The genuine deltas are architectural /
ecosystem, ranked below.

## F1 — eSurge weight-resharding ⭐ FLAGSHIP · effort M · risk high

**Today:** the only weight-swap reshard is `ExecutionManager.update_graphs`
(`libs/easydel/easydel/inference/esurge/runners/execution_manager.py:742`,
block ~`:782-798`): naive per-leaf `jax.device_put` onto
`spx.extract_sharding_structure(template, mesh=self.mesh)` where
`self.mesh == model.mesh`. `esurge_compatible_model`
(`libs/easydel/easydel/infra/mixins/generation.py:5129`) reuses `config.mesh`,
so the "inference mesh" IS the training mesh — swaps only work when the serve
device set == train device set.

**Key asset:** spectrax already ships the general cross-device-set placement
primitive `place_setup_tree_with_shardings` / `place_setup_leaf_with_sharding`
(`libs/spectrax/spectrax/sharding/placement.py:515`/`:419`): same-set →
`device_put`; target ⊆ source → **zero-copy subset rewrap** via
`make_array_from_single_device_arrays` (no host round-trip); disjoint → host
staging. `update_graphs` does NOT use it yet.

**Gap:** no cross-mesh transfer; no distinct serve sharding (TP-only serve vs
FSDP+TP train); transient ~2× weight HBM; `RemoteEngineHandle` has no
`update_model_weights` (DP-replica/decoupled engines can't be synced).

**Design / extension points:**
1. Route `update_graphs` through `spx.place_setup_tree_with_shardings(..., donate=True)` (drop-in for the two `tree_map(device_put)` blocks) → correct for serve ⊆ train (zero-copy) and disjoint (host staging), no new mechanism.
2. Add `esurge_serve_sharding_axis_dims` (`TrainingArguments`, near the other `esurge_*` fields, `training_configurations.py:630-692`) + a `serve_sharding_axis_dims` arg to `get_esurge` (`generation.py:5756`), building a distinct serve mesh via `EasyDeLBaseConfig.create_mesh` (`base_config.py:1217`, 6-axis order). `esurge_compatible_model` uses that clone so `execution_manager.self.mesh` becomes the serve mesh.
3. Add a `stream: bool` path to `place_setup_tree_with_shardings` (spectrax) — place one leaf, drop the source ref — to cap transient HBM.
4. Add `RemoteEngineHandle.update_model_weights(checkpoint_path=..., restart_scheduler=True)` (`inference/esurge/distributed/remote_engine.py:280`) + a wire control-op handled owner-side in `OwnerRequestPlane` (`request_plane.py:213`), syncing cross-runtime replicas via an eformer TensorStore checkpoint (honors `fused_param_tp` retp-on-load).

**Lifecycle already present:** `eSurge.update_model_weights`
(`inference/esurge/esurge_engine.py:2499`) terminates→drains→reshards→
`request_plane.reset()` (`request_plane.py:325`)→rebuilds→re-initiates; the ZMQ
leader is restartable specifically for hot-swaps (`zmq_coordinator.py:262`).

**Layering:** general placement/reshard lives in spectrax; cross-runtime reload
via eformer TensorStore; easydel composes. 6-axis convention preserved.

**TPU reality:** zero-copy subset-rewrap only applies inside ONE JAX runtime
addressing both subsets (single libtpu client). A new serve mesh = new sharding
fingerprint → one-time full bucket recompile; weight refresh within a fixed
serve mesh must NOT recompile — extend the graphdef-fingerprint guard
(`generation.py:5491-5512`) to also key on serve-mesh identity.

**Validation:** core (items 1-3) CPU-verifiable on the fake-8-device mesh (build
eSurge on a 4-device sub-mesh via `serve_sharding_axis_dims`, swap, assert each
leaf's `NamedSharding` + numerical forward parity; both rewrap and host-staging
branches). New `TrainingArguments` field → extend
`tests/trainers/test_training_arguments_save_load_roundtrip.py`. Pod-bound:
cross-runtime reload latency, real HBM during streaming, disjoint multislice.

**Design-lock (before coding):** sharding-expert (mandatory), inference-expert
(recompile + plane lifecycle + cache-shape reuse), jax-expert (cross-mesh
device_put/donation), tpu-expert (single-process two-mesh feasibility).

## F2 — Decoupled rollout on TPU · effort L · risk high · pod-gated

The single libtpu lock forbids two JAX processes on the same chips, so verl-style
disaggregation becomes **checkpoint-mediated across separate JAX runtimes**.
Build on attach-mode `_serve_replicated` (`libs/easydel/easydel/scripts/elarge.py:275`,
env carving `:227`) + eray host dedication (`plan_host_partition`
`libs/eray/eray/resources/topology.py:189`; whole-slice `pod_count`): trainer
drives rollout replicas on dedicated pod hosts, syncs every `weight_sync_steps`
via F1's `RemoteEngineHandle.update_model_weights(checkpoint_path=...)`, plane
routes prompts. Keep F1's subset-rewrap as the co-located fast path. No
Pathways/IFRT exists in-repo — design for per-host multi-controller + ZMQ +
MegaScale.

## F3 — Code-sandbox hardening · effort M · risk medium · ☑ DONE (2026-07)

`CodeVerifier`/`PythonCodeTool`/`BashTool` ran untrusted model code in-process
with only main-thread `signal.alarm` + full parent env. Shipped
`easydel/trainers/_shared/sandbox.py::Sandbox`: child-process execution with
`RLIMIT_CPU`/`RLIMIT_AS`, process-group-kill timeout (works off any thread),
scrubbed env, temp cwd, isolated interpreter (`-I`), optional bwrap/nsjail for
fs/pid/net namespaces. Routed all three call sites through it; tests in
`tests/trainers/test_sandbox.py`.

## F4 — Benchmarked large-scale RL reproductions · effort L · pod-gated

DAPO/Dr.GRPO exist as GRPO `loss_type`s but there are no published, benchmarked
recipes (DAPO/ReTool/Seed-Thinking) or 235B/671B-MoE RL validation. Add eLarge
YAMLs + `examples/post_training/` recipes on top of F1/F2. Recipe+validation
effort, not new infra.

## F5 — Ecosystem gaps · effort S–M · low priority

- Missing example scripts: `rloo`, `papo`, `nemo_gym`, `grpo_replay_buffer`, `gspo_token`.
- Pluggable tracker registry (only wandb + TensorBoard wired,
  `training_configurations.py:1901`/`:2045`); add mlflow/swanlab adapters.
- Niche algos PRIME/VAPO/ReMax/REINFORCE++/GPG → add as GRPO/PPO-family
  `loss_type`s in the existing `_fn.py` (hosts ~9 already), each with a
  loss-math test — NOT new engines.

## What NOT to do

- Do NOT re-integrate vLLM/SGLang instead of eSurge — an external XLA client
  forfeits F1's zero-copy weight path (forces a host/checkpoint round-trip every
  sync) and forks the paged-KV stack. eSurge IS the differentiator.
- Do NOT target AMD/Ascend (upstream-JAX territory).
- Do NOT invent a second resharding mechanism — reuse the spectrax primitive.
- Do NOT add a serve-mesh `TrainingArguments` field without JSON-roundtrip coverage.
- Do NOT assume Pathways/single-controller on TPU.

## Phasing

- P0: design-lock F1 (sharding/inference/jax/tpu) on the CPU-8 mesh.
- P1: F1 items 1-3 (route through the primitive + serve mesh + streamed placement) — CPU-verified, highest value.
- P2: F1 item 4 (cross-runtime sync) — loopback + pod.
- P3: F2 decoupled rollout (pod).
- Parallel/independent: F3 ☑; F5 as capacity allows. Gated on a big pod: F4.

## Cross-cutting risks

- Recompile: distinct serve mesh forces a one-time bucket recompile; steady-state
  refresh within a fixed serve mesh must not — extend the graphdef fingerprint.
- Checkpoint compat: cross-runtime reload must honor `fused_param_tp` (retp-on-load)
  or fused QKV/gate-up scrambles at serve tp>1.
- `extract_sharding_structure(template, mesh=serve_mesh)` must resolve valid specs
  for the serve topology; `resolve_safe_sharding` (`infra/sharding.py:830`) is the fallback.
