# VinePPO Implementation Plan for EasyDeL

This proposal outlines two phased options for bringing true VinePPO-style credit assignment to EasyDeL, building on existing inference engines (`vinference`, `vsurge`, `esurge`), the caching utilities already present in the training/inference stack, and low-level optimisations available via [`ejkernel`](https://github.com/erfanzar/ejkernel).

The plan assumes the current code base (post ProRL/DAPO integration) with GRPO providing GAE/n-step advantages.

---

## Goals

1. **Short-term (2–3 weeks)** — deliver "VinePPO-Lite" (partial MC bootstrapping) for moderate gains without heavy engineering overhead.
2. **Long-term (~3 months)** — full VinePPO with per-token branching rollouts, leveraging cached KV states and optimised kernels for throughput.

Both tracks should share as much infrastructure as possible so the Lite groundwork feeds into the full version.

---

## Shared Foundations

### A. Naming & Documentation Clean-up (immediate)
- Replace "VinePPO advantages" in code/doc comments with "GAE advantages" or "VinePPO-inspired" + caveat.
- Update `RL_ALGORITHMS.md`, `docs/trainers/grpo.md`, and config docstrings to set correct expectations.

### B. Cache Introspection Utilities
- Inventory existing cache tooling:
  - `easydel/layers/caching/`: static/page caches for autoregressive decoding.
  - `vinference`, `vsurge`, `esurge` engines: how they expose `past_key_values` / KV states.
- Expose a unified API from model wrappers (e.g., `EasyDeLElargeModel`) to snapshot KV state at step *t* and resume decoding from that snapshot.
- Ensure the snapshot/resume API is available on both TPU and GPU paths; coordinate with `ejkernel` for fast fused attention kernels when resuming from cache.

Deliverable: `cache_utils.py` with helpers `capture_state(model, prompt_ids, position) -> CachedState` and `resume_from_state(model, cached_state, continuation_ids)`.

### C. Rollout Sampler Harness
- Reuse `vinference` batching logic to produce auxiliary completions in parallel.
- For accelerated throughput, optionally plug into `vsurge`/`esurge` driver so auxiliary samples piggy-back on the high-throughput schedulers.
- Add sampling knobs (K rollouts, max depth, temperature) to a shared sampler API used by both VinePPO-Lite and full VinePPO.

Deliverable: `vineppo_sampler.py` providing `generate_branches(model, cached_states, sampling_params, engine="auto")` that can pick between vanilla generation or a `vsurge`/`esurge` engine when available.

---

## Option A: VinePPO-Lite (Phase 1, 2–3 Weeks)

### Concept
- Avoid per-token branching; instead, run a small number of **auxiliary completions from selected anchor positions** (e.g., final token, optionally mid-sequence checkpoints).
- Use the observed returns to produce Monte Carlo estimates for those anchors and bootstrap standard GAE/n-step advantages across the sequence.

### Steps
1. **Anchor Selection**:
   - Default to only the final token (cheap).
   - Optional config to sample additional anchor indices (e.g., divide sequence into quartiles).

2. **Auxiliary Rollouts**:
   - For each anchor, capture KV state once and launch `K_aux` completions using the shared sampler harness.
   - Reuse caching to avoid redoing prompt prefill.

3. **MC Value Estimation**:
   - Average observed returns across auxiliary rollouts → `V_MC(anchor)`.
   - Interpolate/propagate these values backward via GAE (`adv_estimator="mc_gae"`).

4. **Integration**:
   - Add a new `GRPOConfig` option, e.g., `vineppo_mode="lite"` with knobs: `lite_anchor_stride`, `lite_num_branches`, `lite_branch_max_len`.
   - Extend preprocessing pipeline in `grpo_trainer.py` to call the sampler when the mode is active.

5. **Compute Budget & Optimisations**:
   - Expect ~2–5× training step cost depending on `lite_num_branches`.
   - Use `ejkernel` fused attention and matmul ops during resumed decoding to reduce branch cost.
   - Allow asynchronous branch generation via `vsurge` queue to hide latency.

6. **Validation**:
   - Unit tests around MC value computation.
   - Benchmark on reasoning tasks (e.g., GSM8K subset) to quantify advantage quality vs. baseline.

### Deliverables
- Updated GRPO trainer with VinePPO-Lite mode.
- Config/documentation updates explaining trade-offs.
- Benchmark script comparing baseline GRPO vs. VinePPO-Lite.

---

## Option B: Full VinePPO (Phase 2, ~3 Months)

### Core Requirements
1. **Per-token KV Extraction**:
   - During policy rollouts, store KV state for each token position (or selected positions for pruning).
   - Tight integration with caching layers; ensure memory management is efficient.

2. **Branching Engine**:
   - For each saved state, generate `K` continuations with length up to `branch_max_len`.
   - Use `vsurge` or `esurge` scheduler to manage the high branching factor.
   - Optionally leverage `ejkernel` custom kernels when decoding multiple branches to maximise throughput.

3. **Reward Folding**:
   - Each branch must be scored by the reward pipeline (already present in GRPO).
   - Combine branch returns into unbiased Monte Carlo estimates per token.

4. **Advantage Computation**:
   - Replace learned/value-network-based baselines with `A_t = r_t + γV_MC(s_{t+1}) - V_MC(s_t)`.
   - Support optional variance reduction (e.g., subtract learned baseline if desired).

5. **Trainer Update**:
   - Either extend GRPO or introduce `VinePPOTrainer`.
   - New configuration fields: `vineppo_branches`, `vineppo_max_depth`, `vineppo_engine`, `vineppo_cache_strategy`.

6. **Performance Engineering**:
   - Branching increases compute cost by ~50–100×; mitigate by:
     - Limiting branching to top-n tokens via heuristics.
     - Sharing computations across branches with `ejkernel` fused operations.
     - Scheduling branches asynchronously (prefill queue vs decode queue).

7. **Monitoring & Tooling**:
   - Instrument with `esurge` metrics to track branch counts, MC value variance, compute budget.
   - Add debugging tools to inspect MC estimates vs. ground-truth returns on toy tasks.

### Milestones
- **M1 (4 weeks)**: KV snapshot/resume infrastructure + branch sampler API.
- **M2 (4 weeks)**: Branch generation integrated with reward scoring and MC estimation.
- **M3 (2 weeks)**: Trainer integration, configs, tests.
- **M4 (2 weeks)**: Optimisation pass (use `ejkernel` kernels, scheduler tuning) + documentation.

---

## Dependencies & Collaboration
- **Inference Team**: Align on cache APIs (`vinference`, `vsurge`, `esurge`).
- **Kernel Team**: Coordinate with `ejkernel` for fused attention/matmul on resumed states.
- **Research/Benchmarking**: Define evaluation suites (GSM8K, MathLib, long-form code generation) to demonstrate benefit.
- **Product/Docs**: Update RL documentation once feature flags are in place.

---

## Risks & Mitigations
| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Memory overhead storing per-token KV states | High | Prune branch positions; compress KV caches; stream to disk if needed. |
| Reward evaluation cost | Medium | Batch reward model inference; reuse reward caching if available. |
| Scheduling complexity | Medium | Reuse `vsurge/esurge` scheduler abstractions; add unit/integration tests. |
| Inconsistent terminology | Medium | Update docs before code landing; guard config flags behind feature names. |

---

## Next Steps
1. Land terminology/doco fixes (GAE wording) immediately.
2. Build cache snapshot/resume utilities and branch sampler harness (shared foundation).
3. Implement VinePPO-Lite (Option A) to gather quick wins and real-world feedback.
4. Evaluate resource budget for full VinePPO and decide whether to proceed (Option B).
5. Loop in Erfan and kernel/inference contributors to align on caching and `ejkernel` integration details.

---

Prepared for discussion with the EasyDeL maintainers.
