# Triton GPU Profiling Cheatsheet

Exact commands, env vars, and IR-dump recipes for the workflow in `SKILL.md`.
Triton hides registers/warps, so two questions dominate: *which config won and
why* (autotune logs + `ncu`), and *what did the compiler actually emit* (IR
dumps). Lock GPU clocks before any timing (see the CUDA cheatsheet's `nvidia-smi
-lgc` recipe).

## 1. Timing — `do_bench`, never hand-rolled

```python
import triton, torch
ms = triton.testing.do_bench(
    lambda: my_kernel[grid](...),
    warmup=25, rep=100,
    quantiles=[0.5, 0.2, 0.8],   # median, p20, p80
    return_mode="median",
)
```

`do_bench` warms up, **flushes the L2 cache between reps** (so you don't measure
a hot cache by accident), and returns robust quantiles. For graph-captured
launch-overhead-free timing use `triton.testing.do_bench_cudagraph`.

Sweep across shapes with the report harness:

```python
@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=["M", "N", "K"], x_vals=[...], line_arg="provider",
    line_vals=["triton", "torch"], line_names=["Triton", "torch"],
    ylabel="ms", plot_name="gemm", args={}))
def bench(M, N, K, provider): ...
bench.run(show_plots=False, print_data=True)
```

Always compare against the **fused** torch/library equivalent, not an unfused
baseline.

## 2. Which config won — autotune logs

```bash
TRITON_PRINT_AUTOTUNING=1 python run.py
```

Prints the selected `triton.Config` and the timings it compared. **First thing
to check when "the same kernel got slower"** — it is usually a different cached
config, often because the `@triton.autotune(key=[...])` is missing a shape dim
that changed. Make the `key` cover every dim that should re-trigger tuning.

## 3. What the compiler emitted — IR/PTX dumps

```bash
TRITON_KERNEL_DUMP=1 \           # dump all IR stages
TRITON_DUMP_DIR=/tmp/triton_dump \
MLIR_ENABLE_DUMP=1 \             # MLIR after each pass (or =kernel_name to filter)
LLVM_IR_ENABLE_DUMP=1 \
TRITON_ALWAYS_COMPILE=1 \        # bypass cache so you see fresh IR
  python run.py
```

The compiled artifacts also live under `TRITON_CACHE_DIR` (default
`~/.triton/cache`): `.ttir` (Triton IR), `.ttgir` (**Triton GPU IR** — layouts
and pipelining live here), `.llir` (LLVM), `.ptx`, `.cubin`, and a `.json` with
metadata.

What to look for:
- In **`.ttgir`**: the `#mma` / tensor-core layout on the `tl.dot` result —
  confirms tensor cores are actually used. The `num_stages` pipelining and async
  copies (`async_copy`/`local_load`) appear here too.
- In **`.ptx`** / ptxas output: register usage and **spill** stores/loads
  (`.local` traffic). Spills usually erase a tiling gain.
- For interactive value debugging (not perf): `TRITON_INTERPRET=1` runs the
  kernel in a pure-Python interpreter where you can `print`/breakpoint.

## 4. Hardware counters — `ncu` works on Triton kernels

```bash
ncu --set full -k "my_triton_kernel" -c 1 -o report python run.py
```

Read the same sections as raw CUDA (see the CUDA cheatsheet): **Speed Of Light**
to classify compute- vs memory-bound, **Memory Workload Analysis** for
coalescing/L2 and **bank conflicts**, **Occupancy** and **Source Counters** for
**register spills**. Triton kernel names are mangled — use `-k` with a regex
fragment, or list launches with `--list-kernels` first.

## 5. Intra-kernel hot regions — proton

```bash
proton run.py            # then inspect with proton-viewer
```

Or in-process: `proton.start("name"); ...; proton.finalize()`. Lighter than a
full `ncu` pass for locating which region dominates.

## 6. Reporting checklist

- GPU, Triton + CUDA versions, clock-lock state
- shape, dtype, and the **winning autotune config** (block sizes, `num_warps`,
  `num_stages`)
- `do_bench` median (+ quantiles) baseline vs candidate, identical conditions
- vs a **fused** torch/library baseline, not unfused
- `ncu` SOL %s that named the bound; spill/bank-conflict findings if relevant
- IR-dump evidence when block/stage tuning didn't explain the time (e.g. "tensor
  cores not engaged", "spilling at `num_stages=4`")

## Common mistakes

- Hand-rolled timing (no warmup, no L2 flush) instead of `do_bench`.
- `autotune` `key` missing a changed shape dim → wrong cached config blamed as a
  regression.
- `num_stages` too high → shared-mem-over-budget compile error or silent spills.
- Assuming `tl.dot` used tensor cores without checking the `.ttgir` layout.
- Testing only the aligned tuned shape and shipping a masking-path bug.
- Comparing against an unfused torch baseline to manufacture a speedup.
