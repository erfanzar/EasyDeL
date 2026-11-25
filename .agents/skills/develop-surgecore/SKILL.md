---
name: develop-surgecore
description: Develop, build, test, or debug EasyDeL's native Rust PyO3/maturin host library under libs/surgecore for eSurge. Use for EngineCore, KvCore, SnapshotIndex, HostPayloadBuilder, Detokenizer, surgecore/_native.pyi, cargo/maturin builds, bitwise golden parity tests, or surgecore release sequencing.
---

# Skill: Develop SurgeCore (`libs/surgecore`)

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. `libs/surgecore` is EasyDeL's native Rust extension (`surgecore._native`, built
with PyO3 `abi3-py311` + maturin) that owns eSurge's host state machine and per-step host computation. Only eSurge
imports it — `import easydel` and training paths do not require `surgecore`.

## Package Boundary (CI-Enforced)

- `libs/surgecore` is a standalone workspace member (mirrored to `erfanzar/esurgecore`).
- **Layering rule:** only `easydel` may import `surgecore`; `surgecore` must never import `easydel`, `spectrax`,
  `ejkernel`, `eformer`, or `eray`. Verify with `uv run lint-imports`.
- Do not edit the standalone mirror directly — this monorepo is the source of truth.

## First Reads

Read these before editing Rust or Python stub surfaces:

- `WORKSPACE.md`
- `libs/surgecore/README.md`
- `libs/surgecore/Cargo.toml`
- `libs/surgecore/pyproject.toml`
- `libs/surgecore/src/lib.rs`
- `libs/surgecore/surgecore/__init__.py`
- `libs/surgecore/surgecore/_native.pyi`

Open the module for the surface you are changing:

- Payload builder: `libs/surgecore/src/payload.rs`, `libs/surgecore/src/python/payload.rs`
- Detokenizer: `libs/surgecore/src/python/detok.rs`
- KV core & recurrent snapshots: `libs/surgecore/src/kv/{mod.rs,pool.rs,group.rs,hash.rs,search.rs,snapshot.rs}`,
  `libs/surgecore/src/python/kv.rs`
- Engine & scheduler:
  `libs/surgecore/src/engine/{mod.rs,request.rs,plan.rs,dp.rs,verify.rs,async_spec.rs,record.rs}`,
  `libs/surgecore/src/engine/sched/{mod.rs,schedule.rs,commit.rs,budget.rs,queue.rs,trim.rs}`,
  `libs/surgecore/src/python/engine.rs`

## Native Surfaces & EasyDeL Bindings

| Surface | Rust module | EasyDeL consumer | Role |
|---|---|---|---|
| `HostPayloadBuilder`, `request_distribution_prefix` | `src/payload.rs`, `src/python/payload.rs` | `libs/easydel/easydel/inference/esurge/runners/executors/batch_preparer.py` | Fills preallocated writeable C-contiguous NumPy buffers (`int32`, `float32` for `packed_f32_padded`) in place with the GIL released |
| `Detokenizer`, `DetokenizerResult` | `src/python/detok.rs` | `libs/easydel/easydel/workers/esurge/pipeline/`, `esurge/engine/output_pipeline.py` | In-process batched incremental UTF-8 streaming detokenization over a HuggingFace tokenizer; steady-state emits deltas only, full `accumulated_text` at finish/resync |
| `KvCore`, `SnapshotIndex` | `src/kv/`, `src/python/kv.rs` | `esurge/core/manager.py`, `esurge/core/recurrent_snapshots.py` | Page pool (LRU free list, null page 0), chained page keys, prefix map, full/sliding-window/chunked-local group rules, transactional allocation/rollback, recurrent-snapshot LRU slots |
| `EngineCore`, `Plan`, `Events`, `PLAN_*` flags | `src/engine/`, `src/python/engine.rs` | `esurge/engine/core_driver.py` | Request table, per-DP queues, FCFS/priority scheduler, chunked prefill, preemption, image-span alignment, async placeholders, `set_verify_budget`, `set_step_cost`, async speculative rollback (`spec_slots`, `inflight_drafts`) |

## Editing & Build Contracts

- **GIL-release & error contracts:** `EngineCore` calls copy inputs, release the GIL, then acquire the core mutex. Bad
  arguments raise `ValueError` without mutating state; internal invariant failures raise `InvariantError` and poison
  the core; calls after `shutdown()` raise `CoreClosed`.
- **Delimited `.pyi` stub contract:** `libs/surgecore/surgecore/_native.pyi` has four marker-delimited sections
  (`# ---- begin/end section: detokenization | payload | kv | engine ----`). Edit only inside the matching section when
  updating PyO3 signatures, and keep `libs/surgecore/surgecore/__init__.py` (`__all__`) synchronized.
- **Pure-Rust feature separation:** `Cargo.toml` defines `default = ["python"]` (`pyo3`, `numpy`). Keep
  `src/{payload,kv,engine}` free of Python dependencies so `cargo test --locked --no-default-features` always passes.
- **Build & editable sync:** `libs/surgecore/pyproject.toml` enables `pyo3/extension-module` under
  `[tool.maturin] features` and tracks `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, and `src/**/*.rs` in
  `[tool.uv] cache-keys`. After Rust edits, rebuild via `uv sync` (or `cd libs/surgecore && maturin build --release` to
  produce an `abi3` wheel covering CPython 3.11–3.13). Watch for a stale `libs/surgecore/surgecore/_native.abi3.so`
  shadowing a newly built wheel — replace or remove stale `.so` files when rebuilding.
- **No new `os.environ` knobs:** pass runtime options explicitly through constructor/method arguments.

## Verification

All `surgecore` checks are pure Rust or host-only Python (`@pytest.mark.cpu_ok` — no JAX computation, no accelerator
needed).

Rust formatting, lint, and unit tests (set `PYO3_PYTHON` to a CPython 3.11–3.13 interpreter because `cargo clippy` and
default `cargo test` link `libpython`; add that interpreter's `LIBDIR` to `LD_LIBRARY_PATH` if the test binary cannot
locate `libpython` at runtime):

```bash
cd libs/surgecore
export PYO3_PYTHON="$(which python3)"
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo test --locked --no-default-features
```

Python golden-parity and invariant tests (bitwise parity against `libs/surgecore/tests/data/{host_payload_goldens,kv_goldens,engine_goldens,engine_goldens_ext}.npz`) plus import layering:

```bash
uv run pytest libs/surgecore/tests
uv run lint-imports
```

When changing contracts consumed by eSurge Python wrappers, also run the host `cpu_ok` eSurge tests (and accelerator
eSurge tests when runner/device behavior is affected) per `.agents/skills/debug-esurge/SKILL.md`.

## Release Sequencing

Releases use `scripts/release.sh surgecore <version>` and `scripts/publish.sh surgecore` (see
`.agents/skills/release-workspace/SKILL.md`). Tagging `surgecore-v*` builds and publishes wheels/sdist in CI. Because
`libs/easydel` pins `surgecore` with an exact version pin, publish `surgecore` to PyPI **before** releasing an
`easydel` version that depends on it.

## Routing

- eSurge Python wrappers, runners, multi-replica serving, or accelerator benchmarks:
  `.agents/skills/debug-esurge/SKILL.md`.
- KV-cache tensor layouts and specs: `.agents/skills/debug-easydel-cache/SKILL.md`.
- Workspace release and mirror rules: `.agents/skills/release-workspace/SKILL.md`.
