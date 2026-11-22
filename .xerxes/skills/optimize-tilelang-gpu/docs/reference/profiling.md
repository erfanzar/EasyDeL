# TileLang GPU Profiling Cheatsheet

Exact commands and recipes for the workflow in `SKILL.md`. TileLang lowers to CUDA, so the strategy is: use the built-in
profiler for timing and correctness, read the **generated CUDA** to confirm intent, then drop to `ncu` on that kernel
for ground-truth attribution. Lock GPU clocks before timing (see the CUDA cheatsheet's `nvidia-smi -lgc` recipe).

## 1. Compile, time, and check correctness with the built-in profiler

```python
import tilelang

kernel = tilelang.compile(func, out_idx=[-1])   # JITs to a callable
profiler = kernel.get_profiler()

# correctness vs an independent reference (torch / cuBLAS wrapper)
profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)   # dtype-aware tol

# timing; pass the reference to get both latencies in one run
latency     = profiler.do_bench(warmup=25, rep=100)
ref_latency = profiler.do_bench(ref_program, warmup=25, rep=100)
print(f"tilelang {latency:.3f} ms vs ref {ref_latency:.3f} ms")
```

`do_bench` warms up and times on CUDA events. Always compare against a **real library baseline** (cuBLAS/torch), not
just an unoptimized hand kernel.

## 2. Autotuning — record the winning config

```python
@tilelang.autotune(configs=[{"block_M": 128, "block_N": 128, "block_K": 32,
                             "num_stages": 3, "threads": 128}, ...])
@tilelang.jit(out_idx=[-1])
def build(block_M, block_N, block_K, num_stages, threads):
    ...
    return kernel_func
```

The autotune result carries the chosen config and its measured latency. **Log it.** A later "slowdown" is usually a
different selected config, not a kernel change. Keep the config space no wider than your attribution justifies — a huge
space is slow to tune and overfits one shape.

## 3. Read the generated CUDA — confirm intent

```python
print(kernel.get_kernel_source())     # the emitted CUDA C++
```

Check that:

- `T.gemm` lowered to **tensor-core MMA** (look for `mma`/`wgmma`/`wmma` or the CUTLASS-style fragment ops), not scalar
  loops.
- `T.copy` vectorized (e.g. `float4`) or used **async copy / TMA** (`cp.async` / bulk copy on Hopper).
- shared-memory allocation and any **swizzle** layout matches your
  `T.annotate_layout` / `T.use_swizzle` intent.
- total shared memory across `alloc_shared` (× `num_stages` double-buffering)
  fits the SM budget — overflow shows up as a build failure or a silent fallback.

Save the source and feed it to `ncu`/`nsys` like any CUDA kernel.

## 4. Ground-truth counters — `ncu` on the generated kernel

```bash
ncu --set full -k "<generated_kernel_name>" -c 1 -o report python run.py
```

Read the same sections as raw CUDA (see the CUDA cheatsheet):

- **Speed Of Light** — classify compute- vs memory-bound.
- **Memory Workload Analysis** — coalescing in `T.copy`, L2 hit rate, and **shared-memory bank conflicts** (the direct
  check on whether your layout / swizzle worked).
- **Occupancy** + **Source Counters** — limiter and **register spills** from oversized tiles / `num_stages`.

`do_bench` tells you *that* a config is faster; `ncu` tells you *why* — you need both to optimize deliberately rather
than by lottery.

## 5. Timeline — `nsys`

```bash
nsys profile --stats=true -o timeline python run.py
```

Use when you suspect host/launch overhead dominates rather than the kernel.

## 6. Reporting checklist

- GPU, CUDA + TileLang versions, clock-lock state
- shape, dtype, tile sizes (`block_M/N/K`), `threads`, `num_stages` (the config)
- `do_bench` latency vs the library baseline (cuBLAS/torch), identical conditions
- `assert_allclose` passing, including a boundary (non-tile-aligned) shape
- the `ncu` SOL %s and any bank-conflict / spill findings
- a generated-source note confirming tensor cores / async copy when relevant

## Common mistakes

- Shared memory over budget once `num_stages` double-buffering is counted → build failure or silent fallback.
- Bank conflicts left in place because the kernel "still works" — `ncu` exposes them immediately.
- Assuming `T.gemm` used tensor cores without checking the generated source.
- Comparing only against an unoptimized baseline, never cuBLAS/torch.
- Autotune space so large it overfits one shape and is slow to tune.
- Timing without locked clocks; validating only the tile-aligned shape.
