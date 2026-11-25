---
name: add-ejkernel-kernel
description: Add, modify, benchmark, or autotune an ejkernel operation or backend kernel. Use for Pallas TPU/GPU, Triton, CUDA, CuTe, TileLang, XLA fallback, kernel registry, module operation wrappers, parity tests, or performance claims in libs/ejkernel. If moving EasyDeL core code into ejkernel or adding EasyDeL OperationImpl adapters, also use port-ejkernel-to-easydel-operation.
---

# Skill: Add Or Update An eJKernel Kernel

Specialization of `.agents/skills/run-research/SKILL.md` for creating or extending operations and backend kernels in
`libs/ejkernel/`.

- For optimizing, profiling, or retuning an existing kernel, load `.agents/skills/optimize-ejkernel-kernel/SKILL.md`
  (and the matching backend skill: `optimize-pallas-tpu`, `optimize-triton-gpu`, `optimize-cuda-gpu`,
  `optimize-cute-gpu`, or `optimize-tilelang-gpu`).
- For wiring an ejkernel operation into EasyDeL as an `OperationImpl` adapter, load
  `.agents/skills/port-ejkernel-to-easydel-operation/SKILL.md` after this skill.
- For `ejkernel.ops` executor/cache/selection internals, load `.agents/skills/debug-ejkernel-ops/SKILL.md`.

## First Reads

Read the authoritative source files before editing:

- `WORKSPACE.md` and `libs/ejkernel/pyproject.toml`
- `libs/ejkernel/ejkernel/kernels/_registry.py` (`Platform`, `Backend`, `kernel_registry`, `validate_signatures`,
  AST `del <param>` unsupported-feature checks)
- `libs/ejkernel/ejkernel/modules/base.py` (`detect_platform`)
- `libs/ejkernel/ejkernel/ops/execution/executor.py` and `libs/ejkernel/ejkernel/ops/config/selection.py`
- `libs/ejkernel/docs/maskinfo_guide.md` when the kernel handles masks
- `.agents/skills/optimize-ejkernel-kernel/docs/reference/profiling.md` before any performance claim
- `.agents/skills/optimize-ejkernel-kernel/docs/reference/llo.md` for TPU Pallas structural diagnosis

> Note: `libs/ejkernel/ejkernel/kernels/_registry.py` and `libs/ejkernel/ejkernel/ops/` source code are authoritative
> over `libs/ejkernel/docs/kernel_registry_system.md` and `libs/ejkernel/docs/api/ops.md` (those prose docs lag behind
> `CUTE`, `TILELANG`, `MPS`, and current selection rules).

Open a nearby implementation for the same backend before writing code:
- TPU Pallas: `libs/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/`
- Triton GPU: `libs/ejkernel/ejkernel/kernels/_triton/flash_attention/`
- CUDA C++: `libs/ejkernel/csrc/flash_attention/` + `libs/ejkernel/ejkernel/kernels/_cuda/flash_attention/`
- CuTe DSL: `libs/ejkernel/ejkernel/kernels/_cute/flash_attention/`
- TileLang: `libs/ejkernel/ejkernel/kernels/_tilelang/flash_attention/`

## Platform And Backend Terms

Use the exact enum values from `libs/ejkernel/ejkernel/kernels/_registry.py`:

- `Platform` (implementation family): `XLA`, `PALLAS`, `TRITON`, `CUDA`, `CUTE`, `TILELANG`.
- `Backend` (hardware family): `CPU`, `GPU`, `MPS`, `TPU`, `ANY`.
- Mandatory XLA reference/fallback: `Platform.XLA`, `Backend.ANY`, `priority=0`.
- Never import or call Pallas, Triton, CUDA, CuTe, or TileLang inside `libs/ejkernel/ejkernel/kernels/_xla/`, and never
  register a hardware-specific kernel as `Platform.XLA`.

State the operation ID and every intended `(Platform, Backend)` pair before editing.

## Required Surface & Wiring

For operation `<k>`, keep all layers aligned:

1. **Mandatory XLA fallback:** `libs/ejkernel/ejkernel/kernels/_xla/<k>/` (`_interface.py`, `_xla_impl_fwd.py`, etc.),
   imported in `libs/ejkernel/ejkernel/kernels/_xla/__init__.py`.
2. **Backend kernel(s):** `_pallas/tpu/<k>/`, `_pallas/gpu/<k>/`, `_triton/<k>/`, `_cuda/<k>/` (+ `libs/ejkernel/csrc/<k>/`),
   `_cute/<k>/`, or `_tilelang/<k>/`, imported in `<backend>/__init__.py` so `@kernel_registry.register(...)` executes
   at import time without initializing JAX backends.
3. **Public operation wrapper & config:**
   - `libs/ejkernel/ejkernel/modules/operations/<k>.py` (`Kernel[Cfg, Out]` subclass using `kernel_registry`,
     `heuristic_cfg`, `candidate_cfgs`, and an `Executor(ConfigSelectorChain(...))` with `ConfigCache`,
     `PersistentCache`, and `AutotunePolicy`).
   - `libs/ejkernel/ejkernel/modules/operations/configs.py` (`BaseOperationConfig` subclass).
   - Export in `libs/ejkernel/ejkernel/modules/operations/__init__.py` and `libs/ejkernel/ejkernel/modules/__init__.py`.
   - Keep backend algorithm bodies out of `modules/operations/<k>.py` (wrappers handle shape/metadata prep,
     `shard_map` construction via `create_shard_map_wrapper`, and executor dispatch).
4. **Benchmark spec + shim (CI-enforced):**
   - Add an `OpBenchmarkSpec` entry in `libs/ejkernel/benchmarks/_op_benchmark_registry.py` (`SPECS`).
   - Add a `libs/ejkernel/benchmarks/benchmark_<k>.py` shim calling `run_benchmark("<k>")`.
   - Do **not** drop ad-hoc `.py` scripts into `libs/ejkernel/benchmarks/`; `test_benchmark_registry_coverage.py`
     enforces an exact allowlist.
5. **Tests:** `libs/ejkernel/test/kernels/<backend>/test_<k>.py` and `libs/ejkernel/test/modules/operations/test_<k>.py`.

## Multi-Backend Guide & CI Guardrails

- **Auto-dispatch precedence:** `detect_platform()` (`libs/ejkernel/ejkernel/modules/base.py`) and
  `Executor._prefer_cuda_cfg()` (`libs/ejkernel/ejkernel/ops/execution/executor.py`) prefer `CUTE > CUDA > TRITON > XLA`
  on NVIDIA GPUs and `PALLAS > XLA` on TPU. `TILELANG` is never auto-selected unless `platform="tilelang"` or
  `prefer_tilelang=True`.
- **Callib / FFI bridges:**
  - Pallas TPU: `pl.pallas_call` or `ejkernel.callib.buffered_pallas_call` (`callib/_pallas_call.py`).
  - Triton GPU (CUDA & ROCm): `ejkernel.callib.triton_call` (`callib/_triton_call.py`); guard optional `triton` imports.
  - CUDA C++: sources in `libs/ejkernel/csrc/<k>/`, per-op `_cuda/<k>/_build.py` (or
    `uv run python -m ejkernel.build_cudalib`), registered via `jax.ffi.register_ffi_target(..., platform="gpu", api_version=1)`.
  - CuTe DSL: `ejkernel/callib/_cute_{call,ffi}.py` (TVM-FFI); guard optional `cutlass`/`cute` imports.
  - TileLang: `@T.prim_func` via `ejkernel/callib/_tilelang_{call,ffi}.py` (`out_idx=None`,
    `execution_backend="tvm_ffi"`).
- **CI guardrails (all enforced in `libs/ejkernel/test/`):**
  - `test/kernels/test_backend_autotune_guardrails.py`: **bans** `@triton.autotune`, `triton.autotune`, `@autotune(`,
    `_AUTOTUNE_CACHE`, `autotune_tilelang_ffi`, and `EJKERNEL_*AUTOTUNE*` anywhere under `ejkernel/kernels/`. All
    autotuning lives in `modules/operations/` via `Executor`/`ConfigSelectorChain` or `ejkernel.ops.tuned`.
  - `test/test_import_backend_init.py`: importing any `ejkernel.*` submodule must **never** trigger JAX backend
    initialization (`xb.backends_are_initialized()` must remain `False`).
  - `test/test_benchmark_registry_coverage.py`: every registered algorithm must have a `SPECS` entry and
    `benchmark_<op>.py` shim, with no stray benchmark scripts.
  - `test/kernels/_tilelang/test_honesty.py`: bans leaking JAX compute calls (`_BANNED_JAX_CALLS`) into `_tilelang/`
    interfaces/impls.
  - **`_registry.py` AST `del <param>` trap:** `_get_ignored_params()` treats any `del <param>` inside a registered
    kernel (for parameters not in `_TUNING_PARAM_NAMES`) as declaring `<param>` unsupported and raises
    `EjkernelRuntimeError` whenever a caller passes a non-default value. Use `del <param>` (or
    `__ejkernel_unsupported__`) deliberately for unsupported features — never casually `del` a supported arg.
  - **Silent optional-dep skips:** `libs/ejkernel/test/conftest.py` skips GPU/Triton/CuTe/TileLang tests when optional
    dependencies or hardware are absent. Verify tests actually executed rather than skipped.

## Structure & Correctness Gates

Run grep checks before benchmarking:

```bash
rg -n "pallas|pallas_call|jax\.experimental\.pallas|\bpl\.|triton_call|tilelang_call|cute_call" \
  libs/ejkernel/ejkernel/kernels/_xla/<kernel>
```

The `_xla/` grep must return zero hits. When an operation supports `method="shard_map"`, assert
`"create_shard_map_wrapper" in OpClass.__dict__` and test the `shard_map` path. Always verify:

- `kernel_registry.validate_signatures("<kernel>")`
- `kernel_registry.get("<kernel>", platform=..., backend=...)` for every registered `(Platform, Backend)` pair
- Value parity (and gradient parity for differentiable ops) against an independent `Platform.XLA` / NumPy reference
  across representative and edge shapes, plus dtype and finiteness assertions.

## Configuration, Autotune & Benchmarks

Use `EJKERNEL_AUTOTUNE_POLICY=heuristics` in deterministic tests; use `autotune` only when actively tuning candidates
(see `.agents/skills/debug-ejkernel-ops/SKILL.md` for `ConfigSelectorChain`, `PersistentCache`, and `ejkernel.ops.tuned`).

Benchmark on target accelerator hardware using `libs/ejkernel/benchmarks/benchmark_suite.py` (steady-state after warmup;
measure compile latency separately if relevant). Checked-in files under `libs/ejkernel/benchmarks/baselines/` are
historical artifacts, not an active perf gate.

```bash
# TPU (XLA vs Pallas)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  EJKERNEL_AUTOTUNE_POLICY=heuristics \
  EJKERNEL_BENCH_OPS=<kernel> \
  EJKERNEL_BENCH_PLATFORMS=xla,pallas \
  EJKERNEL_BENCH_CONFIG_LIMIT=1 \
  EJKERNEL_BENCH_WARMUP=5 \
  EJKERNEL_BENCH_ITERS=30 \
  EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py

# GPU (XLA vs GPU backends)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  EJKERNEL_AUTOTUNE_POLICY=heuristics \
  EJKERNEL_BENCH_OPS=<kernel> \
  EJKERNEL_BENCH_PLATFORMS=xla,triton,cuda,cute,tilelang \
  EJKERNEL_BENCH_CONFIG_LIMIT=1 \
  EJKERNEL_BENCH_WARMUP=5 \
  EJKERNEL_BENCH_ITERS=30 \
  EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_gpu \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

Read `.agents/skills/optimize-ejkernel-kernel/docs/reference/profiling.md` for profiling/reporting rules and
`.agents/skills/optimize-ejkernel-kernel/docs/reference/llo.md` for TPU LLO/Mosaic diagnosis.

## Verification Commands

Invoke `uv run pytest` directly from the repo root (do **not** invoke `libs/ejkernel/test/run_tests.py` from repo root —
it passes relative `test/...` paths without `chdir` and lacks `--cute`/`--tilelang`). All kernel/op tests execute JAX
numerics and MUST run on TPU or GPU; without an accelerator, report the change as **unverified on hardware** (check
`.agents/ops/OPS.md` if `libtpu` is busy).

```bash
# Pure-Python CI guardrails (safe without accelerator via CPU trio)
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest \
    libs/ejkernel/test/test_benchmark_registry_coverage.py \
    libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py \
    libs/ejkernel/test/test_import_backend_init.py \
    libs/ejkernel/test/kernels/_tilelang/test_honesty.py

# TPU computation tests (XLA fallback, TPU Pallas, module op)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest \
    libs/ejkernel/test/kernels/_xla/test_<kernel>.py \
    libs/ejkernel/test/kernels/_pallas/tpu/test_<kernel>.py \
    libs/ejkernel/test/modules/operations/test_<kernel>.py

# GPU computation tests (XLA fallback, Triton/CUDA/CuTe/TileLang/Pallas GPU, module op)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run pytest \
    libs/ejkernel/test/kernels/_xla/test_<kernel>.py \
    libs/ejkernel/test/kernels/_<backend>/test_<kernel>.py \
    libs/ejkernel/test/modules/operations/test_<kernel>.py
```

## Definition Of Done

- Every `(Platform, Backend)` implementation is registered, exported, and passes `validate_signatures("<kernel>")`.
- `SPECS` entry + `benchmark_<kernel>.py` shim exist and all static guardrails + `uv run lint-imports` pass.
- Accelerator parity/gradient tests (and benchmarks for perf claims) ran on target TPU/GPU, or the report explicitly
  marks hardware verification as **unverified on hardware**.
- Only the measured winning path is kept.
