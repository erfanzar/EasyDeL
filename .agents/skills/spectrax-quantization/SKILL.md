---
name: spectrax-quantization
description: Work on SpectraX module-path quantization (QAT and narrow-precision quantized training) under libs/spectrax/spectrax/quantization and libs/spectrax/spectrax/nn/_quant.py. Use for QuantProvider, QuantRule, quantize_model, fake_quant, qdot_general, qeinsum, qragged_dot, QArray, narrow_storage_update, stochastic_round, or EasyDeL's apply_quantization_rules wiring.
---

# Skill: Work On SpectraX Quantization

This is a specialization of `.agents/skills/run-research/SKILL.md`. Load and follow `run-research` first.

Use this skill for work inside `libs/spectrax/spectrax/quantization/`, the layer bridge
`libs/spectrax/spectrax/nn/_quant.py`, or debugging QAT / quantized-training numerics, rules, MoE `qragged_dot`, and
narrow-float weight storage.

## Package Boundary

`libs/spectrax` is an independent foundation package (`pyproject.toml` `[tool.importlinter]`). It MUST NOT import
`easydel`, `ejkernel`, `eformer`, `eray`, or `surgecore`. Framework-neutral Qwix-derived numerics live inside
`libs/spectrax/spectrax/quantization/` (no Flax dependency). EasyDeL's preset/config wrapper lives in
`libs/easydel/easydel/layers/quantization/_quantized_training.py`. Verify with `uv run lint-imports`.

## First Reads

- `libs/spectrax/spectrax/quantization/__init__.py`
- `libs/spectrax/spectrax/quantization/_rules.py` (`QuantRule`, `QuantProvider`, `QuantPlan`, `quantize_model`, `unquantize_model`, `rule_for`, `PLAN_ATTRIBUTE`, `DEFAULT_OP_NAMES`, `from_preset`, `from_intmp`)
- `libs/spectrax/spectrax/quantization/_ops.py` (`fake_quant`, `qdot_general`, `qeinsum`, custom VJP + collective calibration)
- `libs/spectrax/spectrax/quantization/_dot.py` (`dot_general`, `loop_dot_general`, `how_to_quantize_for_dot`, `MIN_TILE_SIZE_FOR_OUTPUT_DEQUANT`)
- `libs/spectrax/spectrax/quantization/_ragged.py` (`qragged_dot`, per-expert `[G, 1, N]` scales, padding-row zeroing)
- `libs/spectrax/spectrax/quantization/_qarray.py` (`QArray`, `generic_broadcast_op`, `split_axis`, `tiled_axes`)
- `libs/spectrax/spectrax/quantization/_calibrate.py` & `_numerics.py` (`HowToQuantize`, `calibrate`, `calibrate_across_axis`, `QType`)
- `libs/spectrax/spectrax/quantization/_narrow_storage.py` (`stochastic_round`, `narrow_storage_update`, `scale_loss_for_narrow_gradients`, `suggested_loss_scale`)
- `libs/spectrax/spectrax/nn/_quant.py` (`quantized_dot_general_for`, `quantized_einsum_for`)
- EasyDeL integration (when debugging EasyDeL): `libs/easydel/easydel/layers/quantization/_quantized_training.py`

## Critical Invariants & Footguns

1. **EasyDeL ordering trap — MUST stamp rules BEFORE `model.to_state()`:**
   - `quantize_model` / `ed.apply_quantization_rules(model, ...)` mutates the live module tree by setting
     `PLAN_ATTRIBUTE = "_quant_plan"` (single leading underscore -> stored in `Module._spx_opaque` -> travels inside
     `GraphDef` and changes `Module.structure_hash()` so compile caches re-key cleanly).
   - **NEVER** call `apply_quantization_rules(state.model, ...)` after `model.to_state()`. `EasyDeLState.model` is a
     property that rebuilds a throwaway `Module` from `state.graphdef` on every access, so stamping `state.model`
     silently discards the rules and trains at full precision. Always run:
     ```python
     model = ed.apply_quantization_rules(model, "int8")
     state = model.to_state()
     ```
   - Never rename `_quant_plan` to `_spx_*`: `_spx_`-prefixed attributes are excluded from `GraphDef` export, which
     would quantize eager calls while silently falling back to full precision under `spx.jit`/`spx.grad`.
2. **Two op tiers (`_ops.py`, `_dot.py`):**
   - **Tier 1 (`fake_quant`):** quantize + dequantize with exact identity STE
     (`stop_gradient(reconstructed) + (array - stop_gradient(array))`, avoiding 1-ULP bf16 drift). Works before any
     custom kernel (Pallas, grouped/collective matmul). Simulates quantization error; does **not** improve throughput.
     Supports `calibration_axis_name` + `calibration_transform` for sharded contracted axes (detaches input via
     `stop_gradient` before collective `pmax`/`pmin`, which lack AD rules).
   - **Tier 2 (`qdot_general`, `qeinsum`, `qragged_dot`):** owns the contraction via `jax.custom_vjp` (enabling
     quantized backward passes when `bwd_qtype` is set). `_dot.dot_general` selects **dequantize-on-output** (true
     low-precision dot, e.g. `int8 x int8 -> int32`) only when both operands are quantized affine types and tiles are
     `>= MIN_TILE_SIZE_FOR_OUTPUT_DEQUANT` (128); otherwise uses **dequantize-on-input** (fused by XLA, required for
     weight-only regimes and codebook `"nf4"`).
   - **QT vs QAT distinction:** `int8`, `int4`, `fp8`, `fp8_e4m3b11`, `deepseek_fp8` run narrow-precision GEMMs for
     throughput; `w4a16`, `w8a16`, `nf4`, `mxfp4` are weight-only QAT (no speedup). Neither reduces master-weight HBM.
3. **Rule schema & op vocabulary (`_rules.py`):**
   - `DEFAULT_OP_NAMES = ("dot_general", "einsum", "ragged_dot")`. Do NOT add inert op names that no layer consults
     (guarded by `libs/spectrax/tests/quantization/test_no_inert_op_names.py`).
   - Supported `QType`s: `int2`-`int8` (`int4`/`int8` native), `float8_e4m3fn`, `float8_e4m3b11fnuz` (fastest native
     FP8 on TPU), `float8_e5m2`, `float4_e2m1fn`, `"nf4"`.
   - `module_path` rewrites literal `/` outside `[...]` to `[./]` so MaxText slash configs (`from_intmp`) match
     SpectraX dotted paths directly. Keep `strict=True` in `quantize_model` so zero-match regexes raise loudly.
   - `tile_size` / `bwd_weight_grad_tile_size`: positive `int` = literal tile size; `float` in `(0, 1]` = reciprocal
     tile count (`1 / shard_count`). `_validate_tile_size` rejects subchannel tiles whose scale overhead meets/exceeds
     payload width (`power_of_two_scale=True` accounts for 8-bit E8M0 exponents, e.g., MXFP4 `tile_size=32`).
   - Non-contracted axes are channelwise by default so fused QKV and gate-up projections keep independent column ranges;
     `weight_block_size` enables 2D square blocks (e.g., DeepSeek-V3 `128x128`).
4. **MoE `qragged_dot` (`_ragged.py`):**
   - Uses per-expert channelwise scales `[G, 1, N]` (NOT shared `[1, 1, N]`), gathered per row via `_row_group_ids`.
   - Widens signed `int4` to `int8` (`_ragged_compute_value`) for TPU ragged kernels that reject S4.
   - Explicitly zeros padded rows beyond `sum(group_sizes)` with `jnp.where` (not multiply-by-zero, which preserves
     NaNs in uninitialized padding).
5. **Narrow-float storage (`_narrow_storage.py`):**
   - Applies to narrow **floats** only (`jax.grad` rejects integer parameters). Requires all three mitigations together:
     `scale_loss_for_narrow_gradients` (prefer `eformer.mpric.DynamicLossScale` in production), wide optimizer moments
     via `narrow_storage_update`, and `stochastic_round` with a fresh PRNG key every step. Note that fp8 parameter
     storage still incurs significant convergence degradation vs bf16; narrowing optimizer moments saves more HBM per
     quality point.
6. **Layer bridge (`libs/spectrax/spectrax/nn/_quant.py`):** every matmul layer in `spectrax.nn` should dispatch through
   `quantized_dot_general_for` / `quantized_einsum_for` (returns plain `jax.lax.dot_general` / `jnp.einsum` when
   unstamped).

## Routing

- EasyDeL fused projection TP layouts, post-training quantization & ejkernel wiring:
  `.agents/skills/quantization-layout/SKILL.md`
- eJKernel weight-packing formats & `quantized_matmul`: `.agents/skills/ejkernel-quantization/SKILL.md`
- SpectraX `Module` / `GraphDef` opaque attributes & `nn` layers: `.agents/skills/spectrax-modules/SKILL.md`
- Quantization design & review: `quantization-expert` agent

## Verification

Computation tests MUST run on an accelerator (TPU shown; for GPU use
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as **unverified on
hardware**.

```bash
uv run lint-imports

env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests/quantization/
```

Key suites in `libs/spectrax/tests/quantization/`: `test_rules.py`, `test_no_inert_op_names.py`, `test_ops.py`,
`test_quantize_model.py`, `test_numerics.py`, `test_exact_affine.py`, `test_affine_integration.py`, `test_ragged.py`,
`test_collective_fake_quant_tiles.py`, `test_collective_ragged_tiles.py`, `test_narrow_storage.py`.
