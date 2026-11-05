# surgecore

EasyDeL's native host library for the eSurge inference engine: a Rust crate (PyO3 + maturin) whose extension module
`surgecore._native` runs eSurge's per-step host computation over numpy buffers with the GIL released.

Today it holds `HostPayloadBuilder`, which fills a batch preparer's preallocated staging arrays with one forward
window's step inputs: token and position packing, query start locations and sequence lengths, rank-major DP packing,
page-table rows, the DP-local page check, the v2 KV slot mapping, the positional request distribution and the packed
sampling vectors. Only eSurge's runner imports it; `import easydel` and training do not need it.

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
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
PYO3_PYTHON=$(which python) LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))') \
  cargo test --locked                       # Rust unit tests (with the Python bindings compiled)
cargo test --locked --no-default-features   # the pure-Rust core alone, no Python needed
python -m pytest tests                      # parity against the goldens in tests/data plus invariant tests (CPU)
```

The pytest files carry `pytest.mark.cpu_ok`: they do no JAX computation. `tests/data/host_payload_goldens.npz` holds
inputs and outputs captured from eSurge's former Python step-input builder (its header names the source commit); the
parity test compares the native builder with it bitwise.
