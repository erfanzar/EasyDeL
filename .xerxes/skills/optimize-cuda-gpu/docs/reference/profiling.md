# CUDA GPU Profiling Cheatsheet

Exact commands, `ncu` sections/metrics, and correctness tools for the workflow in `SKILL.md`. Two rules underpin all of
it: **lock clocks before timing**, and **`ncu` is for attribution, CUDA events are for timing** (ncu replay times are
not your benchmark).

## 0. Stabilize the device first

```bash
sudo nvidia-smi -pm 1                     # persistence mode on
nvidia-smi -q -d SUPPORTED_CLOCKS         # list lockable clocks
sudo nvidia-smi -lgc <gfx_MHz>            # lock graphics clock
sudo nvidia-smi -lmc <mem_MHz>            # lock memory clock (newer drivers)
# ... run experiments ...
sudo nvidia-smi -rgc && sudo nvidia-smi -rmc   # reset when done
```

Without this, boost and thermal drift will masquerade as a result.

## 1. Build for profiling

```bash
nvcc -O3 -lineinfo -Xptxas -v kernel.cu -o app
```

- `-Xptxas -v` prints per-kernel **registers/thread, shared mem/block, and spill stores/loads**. Spills are silent in
  source and often the whole story.
- `-lineinfo` lets `ncu` correlate counters back to source lines.
- Test `__launch_bounds__` / `-maxrregcount=N` and re-read `-v` to confirm you cut registers *without* introducing
  spills.

## 2. Timing — CUDA events, not wall clock

```cpp
cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
for (int i = 0; i < WARMUP; ++i) kernel<<<g,b>>>(...);   // warm up
cudaEventRecord(a);
for (int i = 0; i < ITERS; ++i) kernel<<<g,b>>>(...);
cudaEventRecord(b); cudaEventSynchronize(b);
float ms; cudaEventElapsedTime(&ms, a, b); ms /= ITERS;  // report median/min
```

Exclude H2D/D2H copies and the first compile from the timed region.

## 3. Nsight Compute (`ncu`) — per-kernel attribution

```bash
ncu --set full -o report ./app                       # full profile → report.ncu-rep
ncu -k "my_kernel_regex" -c 1 --set full ./app       # one launch of one kernel
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./app
ncu --launch-skip 5 --launch-count 1 ./app           # skip warmup launches
```

Read sections in this order:

1. **GPU Speed Of Light Throughput** (`SpeedOfLight`) — the classifier. Compare **Compute (SM) %** vs **Memory %** of
   peak. Whichever is higher names the bound. Includes the **roofline chart**.
2. **Memory Workload Analysis** (`MemoryWorkloadAnalysis`) — **sectors/request**
   (coalescing: 32 = perfect for a 128-byte line, higher = scattered), L1/L2 hit rates, and **shared-memory bank
   conflicts**.
3. **Occupancy** (`Occupancy`) — achieved vs theoretical and the **limiter**
   (registers / shared mem / block size).
4. **Scheduler / Warp State Statistics** (`SchedulerStats`, `WarpStateStats`) — top **stall reasons** (e.g.
   `Long Scoreboard` = waiting on global memory;
   `MIO Throttle`; `Barrier`).
5. **Source Counters** (`SourceCounters`) — branch efficiency (divergence) and per-line local-memory traffic (spills).

Useful targeted metrics (`ncu --metrics ...`):

```
sm__throughput.avg.pct_of_peak_sustained_elapsed              # compute SOL
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed        # DRAM SOL
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum      # shared bank conflicts
launch__registers_per_thread                                   # reg pressure
smsp__sass_average_branch_targets_threads_uniform.pct          # branch uniformity
```

## 4. Nsight Systems (`nsys`) — timeline / overlap

```bash
nsys profile --stats=true -o timeline ./app    # prints summary tables
nsys-ui timeline.nsys-rep                       # GUI timeline
```

Use to decide whether the kernel is even the bottleneck: launch overhead, serialized H2D/D2H copies, missing stream
overlap, or many tiny kernels (→ fuse / CUDA graphs).

## 5. Correctness — `compute-sanitizer`

```bash
compute-sanitizer --tool memcheck   ./app   # OOB / misaligned global access
compute-sanitizer --tool racecheck  ./app   # shared-memory data races
compute-sanitizer --tool synccheck  ./app   # illegal __syncthreads usage
compute-sanitizer --tool initcheck  ./app   # uninitialized global reads
compute-sanitizer --tool memcheck --leak-check full ./app
```

Run **before** trusting any performance result. A race or OOB makes the timing meaningless.

## 6. Reporting checklist

- GPU model, CUDA/driver version, clock-lock state
- problem size, dtype, launch config (grid/block/dynamic smem)
- registers, shared mem, and spills from `-Xptxas -v`
- the SOL Compute % vs Memory % that named the bound, plus the top stall reason
- event-timed baseline vs candidate (median/min over N iters), identical conds
- `compute-sanitizer` clean
- re-profile after the change showing the targeted stall/limiter shrank

## Common mistakes

- Reporting `ncu` replay times as benchmark numbers.
- Timing without warmup, or including H2D/D2H copies / first compile.
- Not locking clocks.
- Raising occupancy into register spills (check `-v`).
- Ignoring `sectors/request` (uncoalesced access) — the most common real bottleneck.
