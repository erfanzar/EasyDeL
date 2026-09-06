---
name: add-ejkernel-kernel
description: Add, modify, benchmark, or autotune an ejkernel operation or backend kernel. Use for Pallas TPU/GPU, Triton, CUDA, CuTe, TileLang, XLA fallback, kernel registry, module operation wrappers, parity tests, or performance claims in libs/ejkernel. If moving EasyDeL core code into ejkernel or adding EasyDeL OperationImpl adapters, also use port-ejkernel-to-easydel-operation.
---

# Skill: Add Or Update An eJKernel Kernel

This is a specialization of `.xerxes/skills/run-research/SKILL.md`.

Load and follow `run-research` first. This skill adds ejkernel-specific routing and standards.

For work that only optimizes, profiles, retunes, or diagnoses an existing kernel, prefer
`.xerxes/skills/optimize-ejkernel-kernel/SKILL.md`.

For work that ports core operation code from EasyDeL into ejkernel or wires an EasyDeL `OperationImpl` adapter to an
ejkernel operation, load
`.xerxes/skills/port-ejkernel-to-easydel-operation/SKILL.md` after this skill. This skill is not enough by itself for
cross-package porting.

## How To Apply This Skill

1. Load `.xerxes/skills/run-research/SKILL.md`.
2. Read the ejkernel docs and nearby implementation paths below.
3. State the intended operation id and every intended `(Platform, Backend)`
   pair before editing.
4. Keep kernel lifecycle and reporting in `run-research`; keep backend, registry, parity, benchmark, profiling, and dump
   details here.

## First Reads

Read the relevant existing docs before editing:

- `WORKSPACE.md`
- `libs/ejkernel/pyproject.toml`
- `libs/ejkernel/docs/kernel_registry_system.md`
- `libs/ejkernel/docs/ops_system_architecture.md`
- `libs/ejkernel/docs/kernel_implementations.md`
- `libs/ejkernel/docs/test_suite_and_examples.md`
- `libs/ejkernel/docs/api/ops.md` for `Executor`, `ConfigSelectorChain`,
  `ConfigCache`, `PersistentCache`, and autotune flow.
- `libs/ejkernel/docs/maskinfo_guide.md` when the kernel handles masks.
- `docs/reference/profiling.md` before making or validating performance claims.
- `docs/reference/llo.md` when TPU Pallas performance is unclear or shape-dependent compiler/runtime failures appear.

Then open a nearby implementation with the same backend and operation shape. Good TPU Pallas exemplars:

- `libs/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_interface.py`
- `libs/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py`
- `libs/ejkernel/test/kernels/_pallas/tpu/test_gated_delta_rule.py`

## Platform And Backend Are Different

Use the registry terms exactly:

- `Platform` is the implementation family: `XLA`, `PALLAS`, `TRITON`, `CUDA`,
  `CUTE`, or `TILELANG`.
- `Backend` is the hardware family: `CPU`, `GPU`, `TPU`, or `ANY`.
- XLA fallback/reference code is `Platform.XLA`, usually `Backend.ANY`.
- TPU Pallas code is `Platform.PALLAS`, `Backend.TPU`.

Never put Pallas code, Pallas imports, Pallas calls, TPU Mosaic logic, or
`jax.experimental.pallas` usage under `libs/ejkernel/ejkernel/kernels/_xla/`, and never register such code as
`Platform.XLA`.

## Required Shape

For an operation `K`, keep the public surface aligned across:

- XLA fallback/reference: `libs/ejkernel/ejkernel/kernels/_xla/<k>/`
- backend implementation: `libs/ejkernel/ejkernel/kernels/_pallas/tpu/<k>/`,
  `_pallas/gpu/<k>/`, `_triton/<k>/`, `_cuda/<k>`, `_cute/<k>`, or
  `_tilelang/<k>/`
- operation wrapper: `libs/ejkernel/ejkernel/modules/operations/<k>.py`
- operation config: `libs/ejkernel/ejkernel/modules/operations/configs.py`
- tests: `libs/ejkernel/test/kernels/<backend>/test_<k>.py` and, when there is a public wrapper,
  `libs/ejkernel/test/modules/operations/test_<k>.py`

Register backend implementations in `_interface.py` with
`kernel_registry.register(...)` from `libs/ejkernel/ejkernel/kernels/_registry.py`. Use `Platform` and `Backend` values
from that file.

The operation wrapper must be a real `Kernel[Cfg, Out]` class when the operation has a public module API. It must use
the existing ops stack:

- `get_impl(...)` dispatches through `kernel_registry`.
- `run(...)` calls the selected implementation.
- `heuristic_cfg(...)` and `candidate_cfgs(...)` describe config selection.
- `_executor` is an `Executor` using `ConfigSelectorChain`,
  `AutotunePolicy`, `ConfigCache`, and `PersistentCache` when applicable.

The operation wrapper may handle shape packing, metadata conversion, and
`shard_map` wrapper construction. It must not become the backend implementation body for the operation.

## Strict Architecture Gate

Before edits, write down the intended layer for each piece of logic:

- backend algorithm body: backend directory such as `_xla/<k>/` or
  `_pallas/tpu/<k>/`
- backend registration: backend `_interface.py`
- public operation/config/executor/autotune: `modules/operations/<k>.py` and
  `modules/operations/configs.py`
- package adapter, if any: EasyDeL `OperationImpl`

Do not proceed until the shape satisfies all of these rules:

- XLA backend has no Pallas imports or Pallas calls.
- Pallas backend is not registered as `Platform.XLA`.
- public ejkernel APIs dispatch through the operation wrapper and executor, not directly to a copied backend body.
- if `method="shard_map"` can be used, `create_shard_map_wrapper` is defined directly on the operation class.
- public head-sharded helpers route through `_executor(...,
  method="shard_map", mesh=..., in_specs=..., out_specs=...)`.
- unsharded helper paths live on the operation class, for example
  `Class._run_unsharded`, not as module-level `_run_unsharded` bypasses.
- EasyDeL adapters, when present, stay thin and delegate to
  `ejkernel.modules.operations`.

If any of these checks fail, fix the architecture before running benchmarks or claiming the task is done.

## Correctness Standard

Minimum checks:

- Value parity against the XLA/simple reference over a small shape grid.
- Gradient parity for differentiable kernels.
- Shape, dtype, and finite-output assertions.
- Backend-specific skip guards for hardware tests.
- Registry signature compatibility across implementations.

Prefer independent references. Do not compare a kernel to another wrapper that ultimately dispatches to the same code
path.

## Structure Validation Gate

Run grep-level structure checks for new or moved operations:

```bash
rg -n "pallas|pallas_call|jax\.experimental\.pallas|pl\." \
  libs/ejkernel/ejkernel/kernels/_xla/<kernel>

rg -n "create_shard_map_wrapper|method=\"shard_map\"|method='shard_map'" \
  libs/ejkernel/ejkernel/modules/operations/<kernel>.py

rg -n "def _run_unsharded" \
  libs/ejkernel/ejkernel/modules/operations/<kernel>.py
```

The XLA grep must have no hits. If the operation uses `method="shard_map"`, add a focused test that asserts
`"create_shard_map_wrapper" in Class.__dict__`
and exercises the executor shard_map path. If an unsharded helper exists, it should be a class method on the operation
class.

When touching registry wiring, add or run a focused test for:

- `kernel_registry.validate_signatures("<kernel>")`
- direct `kernel_registry.get(...)` lookup for every intended
  `(Platform, Backend)` pair
- parity against a reference that does not dispatch back to the same candidate

## Performance Standard

Benchmark claims require direct numbers:

- previous path or reference vs candidate
- hardware type
- shape/dtype grid
- compile-including timing when relevant
- steady-state timing
- failure cases or narrowed scope

Use the registry-driven benchmark suite first when the operation has a spec. For TPU Pallas work, run the benchmark on
TPU and compare `xla` against
`pallas` on the same device:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
EJKERNEL_BENCH_OPS=<kernel> \
EJKERNEL_BENCH_PLATFORMS=xla,pallas \
EJKERNEL_BENCH_CONFIG_LIMIT=1 \
EJKERNEL_BENCH_WARMUP=1 \
EJKERNEL_BENCH_ITERS=1 \
EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

The suite writes JSON and Markdown reports. The timing from
`ejkernel.benchmarks.Benchmark` is steady-state after warmup; it is not compile-including timing. If compile latency
matters, measure the first JIT call separately and label it separately.

Single-op benchmark wrappers exist under `libs/ejkernel/benchmarks/`, for example:

```bash
uv run python libs/ejkernel/benchmarks/benchmark_gated_delta_rule.py
```

Older fused-loss benchmark examples:

- `libs/ejkernel/scripts/benchmark_fused_losses.py`
- `libs/ejkernel/scripts/bench_final.py`
- `libs/ejkernel/scripts/bench_kl_target.py`

For other kernels, add or extend a focused benchmark under
`libs/ejkernel/benchmarks/` or `libs/ejkernel/scripts/` and keep raw output.

## Configuration And Autotune Routing

When an operation has tunable configs, use the existing ops stack instead of a new ad hoc cache:

- `libs/ejkernel/ejkernel/ops/config/selection.py` for
  `ConfigSelectorChain`.
- `libs/ejkernel/ejkernel/ops/config/cache.py` for `ConfigCache`.
- `libs/ejkernel/ejkernel/ops/config/persistent.py` for `PersistentCache`.
- `libs/ejkernel/ejkernel/ops/execution/executor.py` for `Executor`.
- Existing operation examples under `libs/ejkernel/ejkernel/modules/operations/`
  that read `EJKERNEL_AUTOTUNE_POLICY`.

Use `EJKERNEL_AUTOTUNE_POLICY=heuristics` for deterministic non-tuning checks when a test should not benchmark configs.
Use `autotune` only when the task is explicitly measuring config candidates.

## Dump And Profile Routing

- For benchmark/profile capture and result reporting, read
  `docs/reference/profiling.md`.
- For HLO/LLO/Mosaic dump-driven TPU diagnosis, read
  `docs/reference/llo.md`.
- If a profile or dump requires the TPU and libtpu is busy, stop and route through `.xerxes/ops/OPS.md`; do not
  substitute CPU timing for TPU claims.

## TPU Pallas Notes

- Run Pallas TPU tests only when this process owns the TPU.
- If Mosaic reports alignment or tiling errors, reduce the candidate surface and compare against the direct-load/XLA
  path before adding DMA or async complexity.
- DMA/async only counts when there is measured overlap and a measured win.
- If Pallas is slower than XLA on one fixed shape, dump XLA and Pallas before broad block-size sweeps. Use
  `docs/reference/llo.md`.
- If the TPU is busy, stop TPU validation and say it was not run. CPU/XLA may be used only for host-side preflight such
  as imports, registry shape, simple reference math, or benchmark harness syntax. It does not validate TPU Pallas
  correctness, Mosaic lowering, LLO behavior, DMA/async overlap, or performance.

## Useful Commands

Host-side preflight only:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/ejkernel/test/kernels/_xla

uv run python libs/ejkernel/test/run_tests.py --xla
```

Target TPU validation:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
uv run python libs/ejkernel/test/run_tests.py --pallas -k <kernel>

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
uv run pytest libs/ejkernel/test/modules/operations/test_<kernel>.py
```

If a TPU benchmark cannot start because libtpu is busy, do not report performance. Route through `.xerxes/ops/OPS.md`.

## Definition Of Done

- The implementation is registered and dispatches through the existing registry/operation wrapper.
- Host-side preflight passes where applicable.
- Accelerator checks or benchmarks ran on the target hardware, or the final report clearly says they were not run.
- Performance changes keep only the measured winning path.
