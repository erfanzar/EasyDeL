# ejKernel Benchmarks

The canonical benchmark entry point is:

```bash
.venv/bin/python benchmarks/benchmark_suite.py
```

It loads the kernel registry, benchmarks every registered operation with every
implementation available on the active JAX backend, and writes JSON plus
Markdown reports under `benchmark_results/`.

Useful environment controls:

```bash
EJKERNEL_BENCH_OPS=flash_attention,quantized_matmul
EJKERNEL_BENCH_SKIP_OPS=rwkv7
EJKERNEL_BENCH_PLATFORMS=xla,tilelang
EJKERNEL_BENCH_IGNORE_PLATFORMS=triton
EJKERNEL_BENCH_CONFIG_LIMIT=1
EJKERNEL_BENCH_WARMUP=2
EJKERNEL_BENCH_ITERS=10
EJKERNEL_BENCH_OUTPUT_DIR=benchmark_results
```

Per-operation `benchmark_<op>.py` files are compatibility shims. They delegate
to the shared operation registry runner and should not carry bespoke platform
comparisons or shape suites.
