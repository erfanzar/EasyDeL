# surgecore

EasyDeL's native host library for the eSurge inference engine: a Rust crate (PyO3 + maturin) whose extension module
`surgecore._native` holds eSurge's host-side state and per-step host computation.

It holds:
- `HostPayloadBuilder`, which fills a batch preparer's preallocated staging arrays with one forward window's step
  inputs, with the GIL released: token and position packing, query start locations and sequence lengths, rank-major DP
  packing, page-table rows, the DP-local page check, the v2 KV slot mapping, the positional request distribution and
  the packed sampling vectors.
- `KvCore`, eSurge's KV-cache core: the page pool with its LRU free list and null page 0, the chained page keys and the
  prefix map, the full, sliding-window and chunked-local group rules, transactional allocation with rollback, and
  publication of pages once their KV is computed. eSurge's `CacheManager` is its binding.
- `SnapshotIndex`, the slots of recurrent-state snapshots taken at prefix-cache page boundaries, which let
  linear-attention and SSM models reuse prefix-cache hits.
- `EngineCore`, eSurge's host state machine: the request table, every data-parallel rank's queues, token budget, KV
  core and snapshot index, and the scheduler over them (FCFS or priority, chunked prefill, the long-prefill cap,
  preemption, whole-prompt admission, image-span alignment, snapshot boundaries, async output placeholders, request
  buckets) with synchronous speculative drafts (draft trimming, EAGLE lookahead, acceptance rollback) and the stop
  checks. `schedule()` returns a `Plan`; `commit()` takes the runner's sampled lists, accepted counts and new drafts
  and returns `Events`. Every call releases the GIL.
- The verification budget of synchronous speculative steps (`EngineCore.set_verify_budget`): from a step-time table
  per compiled token bucket (measured by the runner after compile, corrected online from measured step times) and
  per-request acceptance estimates per draft position, each schedule verifies 0..k drafts per request, the count
  that maximises expected emitted tokens per millisecond, and sets `Plan.num_draft_tokens_next`.
- Async speculative steps: a verify window of `1 + k` may be scheduled while the request's previous window is in
  flight. The core advances `computed` as if every draft were accepted (`Plan.spec_slots` per entry), each commit
  rolls back only its own rejected drafts, and the device corrects the later step's start. Windowed groups drop
  pages only below the lowest position the request may really be at.

Only eSurge imports it; `import easydel` and training do not need it.

## Build

```bash
cd libs/surgecore
maturin build --release        # wheel with surgecore/_native.abi3.so (abi3: one wheel covers CPython 3.11-3.13)
```

Install the wheel, or extract its `surgecore/_native.abi3.so` into `libs/surgecore/surgecore/` and put
`libs/surgecore` on `PYTHONPATH`. `pyproject.toml` enables `pyo3/extension-module` through `[tool.maturin] features`,
so plain cargo builds link against libpython.

## Test

```bash
cd libs/surgecore
export PYO3_PYTHON=$(which python)          # a Python 3.11-3.13 interpreter: clippy and the tests link libpython
# (if the test binary cannot find libpython at run time, add that interpreter's LIBDIR to LD_LIBRARY_PATH)
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked                         # Rust unit tests (with the Python bindings compiled)
cargo test --locked --no-default-features   # the pure-Rust core alone, no Python needed
python -m pytest tests                      # parity against the goldens in tests/data plus invariant tests (CPU)
```

Workspace CI (`.github/workflows/workspace-ci.yaml`, job `surgecore`) runs these and a wheel build whenever
`libs/surgecore` changes. Releases go through `scripts/release.sh surgecore <version>` and
`scripts/publish.sh surgecore` (see the workspace's `WORKSPACE.md`); a `surgecore-v*` tag builds and publishes the
wheels and the sdist.

The pytest files carry `pytest.mark.cpu_ok`: they do no JAX computation. The fixtures in `tests/data` hold inputs and
outputs captured from the Python code each piece replaced, and each header names its source commit:
`host_payload_goldens.npz` comes from eSurge's former step-input builder, `kv_goldens.npz` from its former KV core,
`engine_goldens.npz` and `engine_goldens_ext.npz` from its former scheduler (the second with speculative drafts,
runner-output gaps, multimodal page keys, request buckets, the finished set and the structured-output flag).
`test_engine_behaviour.py` states the former scheduler's behaviour tests against `EngineCore`.
The parity tests compare the native code with them bitwise.
