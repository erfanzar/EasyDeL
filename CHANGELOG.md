# Changelog

---

## 2026-03-17 — (unreleased)

### feat: Add loss strategy abstraction, TPU preemption checkpointing, training resume improvements, and license headers

**Loss Strategy Abstraction** — Introduced a two-stage loss protocol (`BaseLossStrategy` / `FunctionalLossStrategy` / `LossForwardPlan`) that separates forward-pass planning from loss computation. The new `CausalLMLossStrategy` tells the model to skip the dense LM-head projection (`apply_lm_head=False`) and instead projects hidden states through the LM head in token-dimension chunks via `causal_lm_loss_chunked_lm_head`, avoiding full `[B, T, V]` logit materialization. Added `resolve_loss_strategy` and a `loss_strategy` cached property on `EasyDeLBaseModule`. Added `compute_lm_logits` to Cohere (logit_scale) and Gemma3 (final_logit_softcapping) models so the chunked path applies model-specific logit transforms.

**Label Smoothing Fix** — Unified label smoothing across all four cross-entropy paths (blockwise, chunked-vocab, chunked-tokens, dynamic) via `_label_smoothing_params` and `_apply_sparse_label_smoothing`. Replaces the previous `eps * mean(logits)` approximation with the exact dense one-hot formulation, matching HuggingFace / Megatron behavior.

**Blockwise CE Improvements** — Wrapped the inner block loop with `jax.checkpoint` to prevent backward-pass memory explosion at long sequence lengths. Made `compute_dtype` configurable (was hardcoded to `float32`). Changed `LossConfig.chunk_block_size` default from `None` to `4096`.

**TPU Preemption Checkpointing** — Full integration of JAX's preemption sync service:
- `EasyDeLPreemptionSignal` control-flow exception
- `_should_save_tpu_preemption_checkpoint` / `_save_tpu_preemption_checkpoint` on `BaseTrainer`
- `save_tpu_preemption_checkpoints` config field (default `True`)
- Sets `JAX_ENABLE_PREEMPTION_SERVICE=true` and `jax_enable_preemption_service` at import time
- Coordinated multi-host save with `sync_global_devices` fencing
- `_prepare_training_output` skips redundant final save when a preemption checkpoint already exists; bare `StopIteration` is now re-raised as `RuntimeError`

**Training Resume Improvements**:
- Epoch-aware step mapping via `_get_epoch_step_bounds` / `_get_resume_epoch`
- `_fast_forward_batches` replays the dataloader iterator by the already-consumed batch count with wraparound
- `_get_next_batch` raises `RuntimeError` on empty dataloaders instead of leaking `StopIteration`
- Removed `step_start_point = step_start_point or 0` coercion so `None` is preserved for auto-resume detection
- Extracted `_save_checkpoint_for_step` to deduplicate checkpointing logic

**LogWatcher** — New `LogWatcher` dataclass and `run_watchers()` for registering arbitrary per-parameter metric functions at configurable intervals. Integrated into `TrainingArguments`, `BaseTrainer`, `BaseTrainerProtocol`, and `eLargeModel.set_watchers()`.

**Distributed Init Hardening** — All three initialization paths (`easydel.__init__`, `infra.init_cluster`, `JaxDistributedConfig.initialize`) now check `jax.distributed.is_initialized()` before attempting init, and re-raise `RuntimeError` only when JAX is genuinely not initialized.

**State Save** — `EasyDeLState.save_state` now accepts an explicit `gather_fns` parameter (forwarded to `save_pretrained`) and writes a `metadata.json` sidecar with step, timestamp, and `is_temporary` flag.

**License Headers** — Added the 2026 Apache 2.0 license header to all 856 Python files under `easydel/` and `tests/`.

**Exports** — Sorted `_import_structure` alphabetically; exported `PreprocessTransform` classes, `LogWatcher`, and `prompt_transforms` from the top-level package.

**179 files changed, +3,950 / -202 lines**

---

## 2026-03-16 — `8aceedf7`

### feat: Add AgenticMoshPit, RLVR, and Embedding trainers with full eLarge support

Massive feature drop introducing three entirely new trainer families and embedding model infrastructure.

**AgenticMoshPit Trainer** — A multi-turn RL trainer with environment interaction. Supports tool-calling (Python code execution, file I/O, web search, regex, shell commands, JSON manipulation), batched rollouts, multiple advantage estimators (GRPO, GiGPO, step_reinforce, agentic_reinforce), and a self-play mode where the model generates its own questions, solves them, and verifies answers. New files: `agentic_moshpit_config.py`, `agentic_moshpit_trainer.py`, `env_manager.py`, `environment.py`, `self_play.py`, `tools.py`, `utils.py`.

**RLVR Trainer** — Reinforcement Learning with Verifiable Rewards for math/code tasks. Uses rule-based reward functions (answer matching, code test execution, format compliance) instead of learned reward models. New files: `rlvr_config.py`, `rlvr_trainer.py`, `reward_verifiers.py`.

**Embedding Trainer** — Contrastive embedding model training with InfoNCE, MNRL, and triplet loss, plus Matryoshka multi-dimensional support. Adds `TaskType.EMBEDDING`, `BaseEmbeddingModule` with configurable pooling strategies (last, first, mean, weighted_mean, max) and L2 normalization, `Qwen2ForEmbedding`, `Qwen3ForEmbedding`, `AutoEasyDeLModelForEmbedding`, `EmbeddingOutput`, and the `encode()`/`cosine_similarity()` convenience API.

Also:

- Auto-detection for reasoning parsers and tool-call parsers based on model type
- 14 new post-training examples using eLarge (SFT, DPO, GRPO, CPO, GFPO, GSPO, KTO, PPO, etc.)
- Removed old tutorial directories, replaced with clean eLarge-based examples
- Fixed `test_generation` sharding bug for models with few KV heads
- Ensured 100% eLarge trainer coverage (cpo, gfpo, gspo were missing)

**128 files changed, +12,986 / -3,129 lines**

---

## 2026-03-15 — `f5c5f652`

### refactor: Rename `elarge_model` to `elarge`, add trainer benchmarking and eSurge eval improvements

- Renamed the entire `easydel/infra/elarge_model` package to `easydel/infra/elarge` with updated imports, docs, and tests
- Split the monolithic `types.py` (1619 lines) into a proper `types/` sub-package with separate modules: `aliases.py`, `data.py`, `engine.py`, `eval.py`, `infra.py`, `model.py`, `quantization.py`, `root.py`, `training.py`
- Removed `normalizer.py` — normalization logic moved into `processing.py`
- Added `maybe_benchmark()` to `BaseTrainer` for running lm-eval benchmarks at configurable intervals during training, with W&B table logging
- Extracted `_esurge_init_kwargs()` helper to deduplicate eSurge engine setup
- Added reasoning boundary token properties (`think_start_token`, `think_end_token`) to eSurge engine
- Improved `eSurgeLMEvalAdapter`: prompt truncation, `chat_template_args` support, empty reasoning scaffold stripping, faithful special-token decoding
- Exported `BenchmarkConfig` from top-level `easydel` package
- New `benchmarking.py` module (547 lines) for integrated eval during training

**61 files changed, +3,964 / -2,387 lines**

---

## 2026-03-14 — `da8dab1e`

### fix: Harden eSurge eval/runtime alignment and serialization

- Rebased runner state and page-table views to active scheduler windows
- Fixed sequence-buffer compaction when multiple holes exist
- Exposed queue budget stats in scheduler output and perf logs
- Normalized lm-eval per-request generation kwargs and stop handling
- Kept greedy evaluation deterministic
- Hidden unfinished reasoning tags from output
- Disabled tokenizer thinking when chat templates support it
- Raised eLargeModel eval defaults and preserved chat templating by default
- Atomically serialized eval results with callable/array support
- Significant model_runner overhaul (+346 lines) for window alignment correctness

**28 files changed, +2,005 / -218 lines**

---

## 2026-03-13 — `b0aea938`

### fix: Ensure dtype fields survive config serialization round-trips

- Added `_coerce_dtype_spec` to rehydrate dtype strings (e.g. `"bf16"`, `"fp8"`, `"fp4"`) back into JAX dtype objects on load
- Handled torch dtype serialization
- Overrode `to_json_string` with dtype-aware normalization including `.to_dict()` objects

**2 files changed, +169 / -2 lines**

---

## 2026-03-12 — `9b45aebe`

### fix: Handle dtype serialization in config and chat-template prompts in generation preview

- Normalized JAX/NumPy dtype-like values to strings when saving config to JSON
- Supported list-of-dict chat prompts in `_prepare_generation_input` for preview generation

**4 files changed, +113 lines**

---

## 2026-03-12 — `c56ec546`

### refactor: Move `auto_remat` from per-sublayer to whole decoder layer and expand gradient checkpoint infrastructure

- Lifted `auto_remat` wrapping from individual attention/MLP sublayers to the full decoder layer block across **all 60+ model architectures** — enables more effective gradient checkpointing
- Added `MLP_NOTSAVEABLE`, `ATTN_NOTSAVEABLE`, and `MLP_ATTN_NOTSAVEABLE` gradient checkpoint policies with regex-based target selection
- Expanded `GRADIENT_CHECKPOINT_TARGETS` with new names (moe_expert_w1/w2/v1, ssm_*, vision_*, projector_*, residual_attn/mlp, etc.)
- Added fractional `epoch_progress` to `StepMetrics` for finer-grained training progress logging
- Added `SFTTrainer._preprocess_batch_input` for assistant_masks → completion_mask conversion and label masking
- Supported 3D `position_ids` in generation (e.g. Qwen VL mRoPE)
- Fixed `SFTConfig` to respect `assistant_only_loss` as fallback for `completion_only_loss`

**80 files changed, +1,070 / -844 lines**

---

## 2026-03-12 — `bcbd9a94`

### refactor: Move GRPO completion chunk size and max loss tokens from env vars to config fields

- Replaced `EASYDEL_GRPO_COMPLETION_CHUNK_SIZE` and `EASYDEL_GRPO_MAX_LOSS_COMPLETION_TOKENS` environment variables with proper `GRPOConfig` dataclass fields (`completion_chunk_size`, `max_loss_completion_tokens`)
- Propagated through `grpo_step` / GFPO trainer call sites

**5 files changed, +44 / -11 lines**

---

## 2026-03-12 — `06dc531c`

### feat: Add multimodal generation support to GRPO/GFPO trainers and upgrade dependencies

- Enabled VLM inputs (`pixel_values`, `image_grid_thw`, `inputs_embeds`, etc.) throughout the generation and scoring pipeline in base trainer, GRPO, and GFPO trainers
- Added `model_kwargs` plumbing to `generate_aio`, `generate_unified`, and compiled generation functions
- Implemented chunked vocab log-prob computation to avoid full-vocab log-softmax materialization
- Supported chunked ref-model log-prob and completion loss computation (`ref_logps_chunk_size` config, `EASYDEL_GRPO_COMPLETION_CHUNK_SIZE` env var) for large batch memory savings
- Fixed VLM generation: strip vision inputs after first step, handle `inputs_embeds` in generation loop, support mRoPE `position_ids`, properly expand multimodal tensors for `num_return_sequences > 1`
- Added GRPO data collator support for multimodal feature keys with left-padding and flattening
- Added `release_generation_runtime()` for explicit HBM reclaim between rollout and scoring phases
- Scoped eSurge cache keys to `EasyDeLState` via UUID for stable cross-reconstruction caching
- Used incremental `wandb.Table` logging for preview generations
- Upgraded Ray 2.53.0 → 2.54.0, eformer 0.0.99.3 → 0.0.99.4
- Added Mistral3 `image_sizes` support in `prepare_inputs_for_generation`
- New `training_utils.py` with 449 lines of multimodal-aware helpers

**22 files changed, +3,043 / -1,357 lines**

---

## 2026-03-10 — `81fb1b64`

### fix: Fix XLA trigger for gated delta rule

- Fixed conditional logic for XLA platform detection in the gated delta rule kernel dispatch

**1 file changed, +3 / -1 lines**

---

## 2026-03-09 — `c14e60a8`

### feat: Add linear attention conv-state helpers, TP-safe ragged cache dtype, num_rows limiting, and quantization group-size fallback

- Added `linear_attention` module with conv-state and masking utilities for KimiLinear, refactored KimiLinear modeling to use shared helpers
- Added TP-compatible dtype auto-selection for v3 ragged page cache storage
- Supported `num_rows` limit on data sources (streaming `.take()` and in-memory `.select()`)
- Added `_effective_ejkernel_group_size` fallback with logging when configured `group_size` doesn't divide the weight shape
- Fixed `craft_sharding` to sanitize `PartitionSpec` rank for scales/biases
- Fixed KDA single-step decode dtype promotion and BTHD layout equivalence
- Added Qwen3Next grouped GDR decode with proper output dtype casting
- Added `RMSNorm`/`LayerNorm` with configurable `zero_centered_gamma`

**35 files changed, +2,090 / -412 lines**

---

## 2026-03-08 — `40330cba`

### docs: Add comprehensive docstrings across the codebase and bump eformer/ejkernel

- Added module-level, class, and method docstrings throughout all major subsystems: data pipeline, inference engine, model configs, trainers, utilities, workers, and scripts
- Bumped eformer to 0.0.99.3 and ejkernel to 0.0.72
- Added docstrings to Qwen3OmniMoE (+439 lines), logits_process (+219 lines), roberta (+160 lines), llama4 config (+146 lines), and many more

**126 files changed, +4,339 / -285 lines**

---

## 2026-03-06 — `94a1a5e4`

### refactor: Replace assert statements with proper exceptions, fix typos, and centralize generation config

- Replaced `assert` statements with `ValueError`/`TypeError`/`RuntimeError` across caching, inference, infra, trainers, and operations modules (112 files)
- Fixed typos: `LOSS_FN_VARIENTS` → `VARIANTS`, `_gready` → `_greedy`, `max_lenght` → `max_length`, `use_data_collactor` → `use_data_collator`
- Centralized generation config wiring (`top_p`, `top_k`, `temperature`, etc.) into `TrainingArguments.__post_init__` instead of duplicating in each trainer config
- Extracted `_apply_training_args_legacy_aliases` and `_parse_partition_spec` helpers into `training_configurations` module
- Fixed `min_p` sampling to use threshold-based filtering instead of cumulative probs
- Fixed `LightningCacheView.concatenate_to_cache` to raise `NotImplementedError`
- Fixed `ConstantList` method signatures (`insert`/`pop`) to match list protocol
- Added proper logging for cache read failures instead of bare `except: pass`
- Bumped eformer to 0.0.99.1
- Changed default `blocksize_k`/`blocksize_q` from 128 to 512
- Fixed `eval_step` to avoid mutating batch dict

**112 files changed, +1,021 / -1,108 lines**

---

## 2026-03-05 — `ddef8f84`

### fix: Reduce HBM spikes in distillation trainer and fix sharding step type

- Extracted `_per_token_xent` helper to reduce peak vocab-sized f32 tensors from 3x to 2x — teacher logits processed first so their scaled intermediates can be freed before student intermediates are materialized
- Kept teacher hidden states/attentions as tuples instead of stacking into dense tensors (avoids a full copy)
- Deleted `teacher_outputs` early to free memory sooner
- Ensured `state.step` is a `jax.Array` before sharding to avoid type errors
- Used `logger.warning_once` instead of `warnings.warn` for flops calculation failure

**3 files changed, +54 / -42 lines**

---

## 2026-03-05 — `b2b5ab4f`

### feat: Add MLA ragged paged attention support and auto-detection for MLA architectures

- Introduced `multi_latent_ragged_page_attention_v1` mechanism with `MLARaggedPagesCache`, `MLARaggedPagesCacheConfig`, and `MLARaggedPagesCacheView` (811 new lines in kernel)
- Added MLA auto-detection in eSurge engine and generation mixin to automatically force MLA-compatible attention kernels for DeepSeek-style models
- Added `mla_attn_mechanism`, `mla_attn_dtype`, `mla_attn_softmax_dtype` config fields to `EasyDeLBaseConfig` for per-layer MLA overrides
- Supported mixed MLA/non-MLA attention layers with separate cache configs
- Updated DeepSeek v2/v3, GLM4 MoE Lite, and GLM MoE DSA models for MLA attention
- Renamed trainer prefixes to PascalCase (e.g., `dpotrainer` → `DPO`)
- Bumped ejkernel to 0.0.71
- Added comprehensive distillation loss math tests (270 lines)

**32 files changed, +2,626 / -199 lines**

---

## 2026-03-04 — `754c44d6`

### fix: Fix distillation trainer and bump eformer to 0.0.98

- Fixed issues in `distillation_trainer/_fn.py` — improved loss computation logic (+57 / -31 lines of reworked code)
- Bumped eformer dependency to 0.0.98

**4 files changed, +57 / -31 lines**

---

## 2026-03-04 — `2b1f3cf8`

### refactor: Delegate GDR kernel to ejkernel module API and improve W&B naming

- Replaced inline gated delta rule kernel (~1,000 LOC removed) with `ejkernel.modules` API — the entire `_ChunkGatedDeltaRuleKernel` class, executor, dispatch function, and forward implementation were deleted in favor of the upstream `GatedDeltaRuleOp`
- Bumped ejkernel to 0.0.70
- Removed `_single_step_gated_delta_rule_fwd` bridge; `qwen3_next` now uses `GatedDeltaRuleOp` directly with BTHD layout
- Added structured W&B run names via `build_wandb_run_name()` with trainer, batch size, learning rate, and optimizer tokens
- Removed model name from W&B project name for stable project grouping
- Renamed `trainer_prefix` defaults to PascalCase (e.g. `sfttrainer` → `SFT`)
- Increased `weight_distribution_log_steps` default from 50 to 500
- Added docs for on_policy_distillation, seq_kd, sparse_distillation trainers and glm_moe_dsa, qwen3_5, qwen3_5_moe modules

**70 files changed, +442 / -1,233 lines**

---

## 2026-03-03 — `fdd0fc1f`

### fix: Fix backward pass for gated delta rule

- Major rework of the gated delta rule kernel backward implementation (+663 / -78 lines), fixing gradient computation correctness

**1 file changed, +663 / -78 lines**
