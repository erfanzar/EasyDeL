# EasyDeL / eformer / ejkernel environment flags

This document lists environment variables that are read or set by EasyDeL, eformer, and ejkernel.

Conventions:

- `check_bool_flag` treats these values as true: `true`, `yes`, `ok`, `1`, `easy` (case-insensitive).
- Defaults below are the values used by the code when the env var is unset.

## EasyDeL startup auto-tuning (EASYDEL_AUTO)

`EASYDEL_AUTO` (default: `true`) controls whether EasyDeL applies startup defaults on import.
Set `EASYDEL_AUTO=0` to skip all of the environment changes below.

Always set when `EASYDEL_AUTO` is true:

- `NUMEXPR_NUM_THREADS=8` (reduce numexpr thread count)
- `KMP_AFFINITY=noverbose` (silence KMP affinity logging)
- `GRPC_VERBOSITY=3`
- `GLOG_minloglevel=3`
- `CACHE_TRITON_KERNELS=1` (cache Triton kernels)
- `TPU_MIN_LOG_LEVEL=3`
- `TPU_STDERR_LOG_LEVEL=3`
- `TPU_LOG_DIR=disabled`
- `TF_CPP_MIN_LOG_LEVEL=3`
- `JAX_ENABLE_PGLE=true`
- `JAX_PGLE_PROFILING_RUNS=3`
- `JAX_PGLE_AGGREGATION_PERCENTILE=85`
- `NCCL_LL128_BUFFSIZE=-2`
- `NCCL_LL_BUFFSIZE=-2`
- `NCCL_PROTO=SIMPLE,LL,LL128`

Appended to existing values when `EASYDEL_AUTO` is true:

- `XLA_FLAGS` adds:
  - `--xla_gpu_triton_gemm_any=true`
  - `--xla_gpu_enable_while_loop_double_buffering=true`
  - `--xla_gpu_enable_pipelined_all_gather=true`
  - `--xla_gpu_enable_pipelined_reduce_scatter=true`
  - `--xla_gpu_enable_pipelined_all_reduce=true`
  - `--xla_gpu_enable_reduce_scatter_combine_by_dim=false`
  - `--xla_gpu_enable_all_gather_combine_by_dim=false`
  - `--xla_gpu_enable_reduce_scatter_combine_by_dim=false`
  - `--xla_gpu_all_gather_combine_threshold_bytes=33554432`
  - `--xla_gpu_reduce_scatter_combine_threshold_bytes=33554432`
  - `--xla_gpu_all_reduce_combine_threshold_bytes=33554432`
  - `--xla_gpu_multi_streamed_windowed_einsum=true`
  - `--xla_gpu_enable_latency_hiding_scheduler=true`
  - `--xla_gpu_enable_cublaslt=true`
  - `--xla_gpu_enable_cudnn_fmha=true`
  - `--xla_gpu_force_compilation_parallelism=4`
  - `--xla_gpu_enable_shared_constants=true`
  - `--xla_gpu_enable_triton_gemm=true`
  - `--xla_gpu_enable_command_buffer=''`
  - `--xla_disable_hlo_passes=collective-permute-motion`
- `LIBTPU_INIT_ARGS` adds:
  - `--xla_tpu_enable_latency_hiding_scheduler=true`
  - `--xla_enable_async_collective_permute=true`
  - `--xla_tpu_enable_ag_backward_pipelining=true`
  - `--xla_tpu_enable_data_parallel_all_reduce_opt=true`
  - `--xla_tpu_data_parallel_opt_different_sized_ops=true`
  - `--xla_tpu_enable_async_collective_fusion=true`
  - `--xla_tpu_enable_async_collective_fusion_multiple_steps=true`
  - `--xla_tpu_overlap_compute_collective_tc=true`
  - `--xla_enable_async_all_gather=true`
  - `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true`
  - `--xla_tpu_megacore_fusion_allow_ags=false`
  - `TPU_MEGACORE=MEGACORE_DENSE`

Set only if missing when `EASYDEL_AUTO` is true:

- `XLA_PYTHON_CLIENT_MEM_FRACTION=1.0`
- `JAX_TRACEBACK_FILTERING=off`

## ejkernel startup defaults

These are set during ejkernel import or profiler setup:

- `TF_GPU_ALLOCATOR=cuda_malloc_async` (forces the CUDA async allocator)
- `TF_CPP_MIN_LOG_LEVEL=3` when the ejkernel profiler is constructed with `silence_tf_cpp_logs=True`
  (default) and the env var is not already set

## Distributed init

| Env var                   | Default | What it does                                                              | Use case                                                                                               |
| ------------------------- | ------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `ENABLE_DISTRIBUTED_INIT` | `true`  | Auto-calls `eformer.executor.DistributedConfig().initialize()` on import. | Set to `0` if you manage JAX distributed init manually or want to avoid auto-init in worker processes. |

## Mesh and sharding

| Env var                           | Default | What it does                                                                                                  | Use case                                                                 |
| --------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `EFORMER_CREATE_MESH`             | `true`  | When true, use eformer mesh creation (mesh_utils/hybrid path). When false, use `jax.make_mesh` when possible. | Flip to `0` to prefer pure JAX mesh creation.                            |
| `MEGASCALE_NUM_SLICES`            | auto    | Overrides detected TPU slice count in mesh creation.                                                          | Use on multi-slice TPU setups when device introspection is insufficient. |
| `MIN_SHARDING_SIZE`               | `16384` | Arrays smaller than this stay unsharded in sharding utilities.                                                | Avoids overhead for small tensors.                                       |
| `LOG_SHARDING_MOVE`               | `false` | Logs warnings when sharding specs are auto-corrected.                                                         | Debug unexpected sharding adjustments.                                   |
| `ED_DEFAULT_HARDWARE_ABSTRACTION` | `false` | Sets default `hardware_abstraction=True` in base config.                                                      | Enable custom hardware abstraction and kernels by default.               |
| `EKERNEL_OPS`                     | `false` | Declared but not currently consumed in-tree.                                                                  | Reserved for external or future kernel-op selection.                     |

## Compilation cache

| Env var                   | Default                                | What it does                                                                             | Use case                                                 |
| ------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `EASYDEL_CACHE_COMPILES`  | `false`                                | Enables persistent caching of compiled JAX functions to disk (ejit in EasyDeL/ejkernel). | Speed up repeated runs.                                  |
| `EASYDEL_RECOMPILE_FORCE` | `false`                                | Forces recompilation even if cache exists.                                               | Debug or refresh cached artifacts.                       |
| `ALLOW_FULL_CACHE`        | `false`                                | Enables full JAX persistent cache of XLA artifacts in ejkernel ejit.                     | Try only if you need full cache; may be unstable on GPU. |
| `COMPILE_FUNC_DIR`        | `${CACHE_DIR}/ejit_compiled_functions` | Directory used to store compiled function cache (EasyDeL/ejkernel).                      | Redirect cache to a fast disk or shared location.        |

## ejkernel autotuning and profiling

| Env var                         | Default                       | What it does                                                                                             | Use case                                                    |
| ------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `EJKERNEL_AUTOTUNE_POLICY`      | `autotune`                    | Cache-miss policy: `autotune` or `heuristics` (some ops default to heuristics).                          | Skip autotune for faster startup or deterministic behavior. |
| `EJKERNEL_LOG_AUTOTUNE`         | `0`                           | Logs candidate timing and best-config selection during autotune.                                         | Inspect tuning decisions and regressions.                   |
| `EJKERNEL_PERSISTENT_CACHE_DIR` | `~/ejkernel-presistent-cache` | Root directory for per-op JSON config caches (fallbacks to `./.ejkernel-presistent-cache` or `$TMPDIR`). | Persist tuned configs across runs or share across machines. |
| `EJKERNEL_OPS_RECORD`           | `0`                           | Records kernel invocations for offline autotuning.                                                       | Enable before running `autotune_lowered`.                   |
| `EJKERNEL_OPS_STAMP`            | `none`                        | Adds profiling labels: `hash`, `json`, or `none`.                                                        | Trace ops in HLO/XProf with readable labels.                |
| `EJKERNEL_OPS_PREFIX`           | `ejkernel_ops#`               | Prefix for stamped operation labels.                                                                     | Customize label namespace to avoid collisions.              |

## ejkernel kernel selection limits

| Env var                       | Default  | What it does                                                  | Use case                                                  |
| ----------------------------- | -------- | ------------------------------------------------------------- | --------------------------------------------------------- |
| `EJKERNEL_TRITON_SMEM_LIMIT`  | `101376` | Shared-memory ceiling (bytes) for Triton candidate filtering. | Lower to avoid SMEM-heavy configs; raise to widen search. |
| `EJKERNEL_RDA_MAX_CANDIDATES` | `32`     | Maximum candidate configs for ragged decode attention (GPU).  | Reduce autotune time or cap search space.                 |

## Training and data

| Env var                | Default | What it does                                                                     | Use case                                          |
| ---------------------- | ------- | -------------------------------------------------------------------------------- | ------------------------------------------------- |
| `FAST_COMPILE`         | `true`  | Skips NaN-guarded updates in `update_state_respectfully` for faster compilation. | Faster compile, but less safety around NaNs.      |
| `SCAN_TRAINER`         | `true`  | Defined as a global flag, not referenced in-tree.                                | Reserved for scan-based trainer workflows.        |
| `HFDATASOURCE_NONSTOP` | `1`     | If `1`, resets iterators when a dataset shard is exhausted.                      | Keep training from stopping on iterable datasets. |
| `TQDM_NCOLS`           | `0`     | Sets tqdm progress bar width; `0` means auto.                                    | Fixed-width progress bars in narrow terminals.    |

## Inference and eSurge

| Env var                           | Default         | What it does                                                              | Use case                                          |
| --------------------------------- | --------------- | ------------------------------------------------------------------------- | ------------------------------------------------- |
| `EASYDEL_TOPK_FOR_COMPUTE`        | `64`            | Limits top-k in efficient top-p sampling.                                 | Trade accuracy vs speed in sampling.              |
| `EASURGE_MAX_SCHEDULER_ERRORS`    | `1`             | Max consecutive scheduler errors before eSurge stops.                     | Increase for resiliency in flaky environments.    |
| `EASURGE_TOKENIZER_ENDPOINT`      | none            | Overrides tokenizer worker ZeroMQ endpoint.                               | Run tokenizer/detokenizer on custom endpoints.    |
| `EASURGE_DETOKENIZER_ENDPOINT`    | none            | Overrides detokenizer worker ZeroMQ endpoint.                             | Same as above.                                    |
| `EASURGE_SYNC_INPUTS_FOR_TIMING`  | `0`             | If `1`, syncs inputs for more accurate timing (adds a device round-trip). | Benchmarking accurate prep time.                  |
| `ESURGE_WORKER_TRUST_REMOTE_CODE` | `1`             | Controls `trust_remote_code` for tokenizer workers.                       | Disable to avoid executing remote tokenizer code. |
| `OPENAI_API_KEY`                  | none            | Used by OpenAI-compatible proxy when `api_key` not provided.              | Required for proxying to OpenAI.                  |
| `PYTHONHASHSEED`                  | `1524618910112` | Used to derive hash seed for prefix caching in eSurge.                    | Make prefix caching deterministic across runs.    |
| `HF_HUB_ENABLE_HF_TRANSFER`       | none            | Set to `1` in download scripts to enable hf_transfer.                     | Faster HF downloads in scripts.                   |

## MoE, kernels, and quantization

| Env var                       | Default | What it does                                                                             | Use case                                                    |
| ----------------------------- | ------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `FORCE_NATIVE_RUNTIME`        | `false` | Forces `forward_native` path for operations, bypassing backend-specific implementations. | Debug or avoid backend-specific kernels.                    |
| `DISABLE_MOE_AUTOTUNE_ON_TPU` | `false` | Disables Pallas GMM autotune on TPU, uses fixed block sizes.                             | Avoid invalid tiling configs or reduce compile variability. |
| `EP_DISPATCH`                 | `auto`  | Global MoE expert dispatch policy (not referenced in-tree).                              | Reserved for external MoE routing control.                  |
| `EP_AUTO_TRESHOLD`            | `0`     | Threshold for automatic expert dispatch (not referenced in-tree).                        | Reserved.                                                   |
| `PERMITTED_KV_KERNELS`        | `true`  | Allows TPU KV-cache update kernels in ragged cache.                                      | Disable to force JAX fallback kernel.                       |
| `USE_NF4_KERNEL_TPU`          | `1`     | Enables NF4 TPU kernels when truthy.                                                     | Set to `0` to force non-kernel path.                        |
| `WARN_ON_MATTER`              | `true`  | Warns when implicit array ops fall back to materialization.                              | Debug missing primitive handlers.                           |

## ejkernel debugging and tests

| Env var                               | Default | What it does                               | Use case                             |
| ------------------------------------- | ------- | ------------------------------------------ | ------------------------------------ |
| `EJKERNEL_MASK_DEBUG`                 | `0`     | Enables MaskInfo debug traces to stdout.   | Inspect attention mask conversions.  |
| `EJKERNEL_RUN_PREFILL_PAGE_ATTENTION` | `0`     | Enables slow prefill page attention tests. | Run optional ejkernel test coverage. |

## Tensor transfer and interop

| Env var                           | Default | What it does                                                   | Use case                           |
| --------------------------------- | ------- | -------------------------------------------------------------- | ---------------------------------- |
| `EASY_SAFE_TRANSFER`              | `true`  | Uses `jax.device_get` + CPU copy for JAX->Torch conversion.    | Safe transfers without GPU DLPack. |
| `EASYDEL_FORCE_TORCH_USE_CPU`     | `false` | Forces CPU path even when CUDA is available.                   | Debug or avoid GPU interop issues. |
| `EASYDEL_PERFRED_HOST_COPY`       | `cpu`   | Target device platform for staging transfers; `none` disables. | Select CPU/GPU/TPU staging device. |
| `EASYDEL_PERFRED_HOST_COPY_INDEX` | `0`     | Device index to use for staging transfers.                     | Choose a specific local device.    |

## Logging

| Env var            | Default | What it does                                         | Use case                                    |
| ------------------ | ------- | ---------------------------------------------------- | ------------------------------------------- |
| `LOGGING_LEVEL_ED` | `INFO`  | Sets log level for EasyDeL/eformer/ejkernel loggers. | Increase to `DEBUG` during troubleshooting. |

## Storage

| Env var              | Default    | What it does                                           | Use case                                |
| -------------------- | ---------- | ------------------------------------------------------ | --------------------------------------- |
| `EASYDEL_GCS_CLIENT` | none       | Path to GCS credentials used by `eformer.paths.ePath`. | Access GCS-backed paths.                |
| `LOCALAPPDATA`       | OS default | Windows cache root used for EasyDeL cache dir.         | Override cache dir location on Windows. |

## Worker process defaults

These are set in worker launchers (`workers/*/worker_manager.py` and `workers/*/worker_main.py`).

| Env var                         | Default | What it does                                | Use case                                        |
| ------------------------------- | ------- | ------------------------------------------- | ----------------------------------------------- |
| `JAX_PLATFORMS`                 | `cpu`   | Forces worker processes to use CPU backend. | Avoid GPU contention in tokenizer/aux workers.  |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Disables GPU memory preallocation.          | Prevent workers from grabbing large GPU memory. |
| `PYTHONUNBUFFERED`              | `1`     | Unbuffered stdout/stderr.                   | Real-time worker logs.                          |
| `ENABLE_DISTRIBUTED_INIT`       | `0`     | Disables auto distributed init in workers.  | Keep workers lightweight.                       |

## eformer executor and Ray

| Env var                        | Default   | What it does                                                  | Use case                             |
| ------------------------------ | --------- | ------------------------------------------------------------- | ------------------------------------ |
| `RAY_ADDRESS`                  | none      | Connects to an existing Ray cluster.                          | Use a remote Ray head node.          |
| `RAY_EXECUTABLE_PATH`          | none      | Overrides the `ray` executable path in TPU patcher.           | Use a custom Ray install.            |
| `COORD_PORT`                   | `8081`    | MegaScale coordinator port for multi-slice.                   | Avoid port conflicts.                |
| `EFORMER_SUBPROCESS_TIMEOUT_S` | `1000000` | Timeout for isolated subprocess calls.                        | Prevent hung subprocesses.           |
| `EFORMER_SCALE_POLL_S`         | `30`      | Poll interval during resource scaling.                        | Slow down scaling checks.            |
| `EFORMER_SCALE_ADD_TIMEOUT_S`  | `604800`  | Timeout for adding resources (seconds).                       | Extend or shorten provisioning time. |
| `EFORMER_SCALE_RETRY_SLEEP_S`  | `60`      | Sleep time between scale retries.                             | Backoff after preemption.            |
| `EFORMER_HOST_HEALTH_WAIT_S`   | `60`      | Wait for hosts to become healthy before dispatch.             | Increase for slow startups.          |
| `EFORMER_SAFE_GATHER`          | `1`       | Prunes dead slice actors before gather.                       | Safer multi-slice execution.         |
| `EFORMER_MODERATE`             | `1`       | Adjusts discovered host/device counts to available resources. | Avoid over-allocating in Ray.        |
| `EFORMER_KILL_VFIO`            | `1`       | Kills processes holding `/dev/vfio/*` before running.         | Free TPU VFIO devices.               |
| `EFORMER_INSTALL_LSOF`         | `0`       | Attempts noninteractive install of `lsof` for VFIO cleanup.   | Enable if `lsof` is missing.         |
| `EXPECTED_TPU_COUNT`           | `64`      | Expected TPU count in patcher sanity checks.                  | Validate TPU pool size.              |

## eformer TPU and MegaScale env vars (set or propagated)

These are typically set by eformer and passed to workers; you usually do not set them manually unless
integrating with custom launchers.

| Env var                         | Default | What it does                                         |
| ------------------------------- | ------- | ---------------------------------------------------- |
| `EXECUTOR_CALL_INDEX`           | none    | Host index within a slice (0-based).                 |
| `EXECUTOR_CALL_SLICE`           | none    | Slice index in multi-slice execution.                |
| `MEGASCALE_COORDINATOR_ADDRESS` | none    | `IP:port` of MegaScale coordinator.                  |
| `MEGASCALE_NUM_SLICES`          | none    | Total number of slices in the pod.                   |
| `MEGASCALE_PORT`                | none    | MegaScale coordinator port.                          |
| `MEGASCALE_SLICE_ID`            | none    | Slice ID for this worker.                            |
| `TPU_SLICE_NAME`                | none    | TPU slice name.                                      |
| `TPU_HOST_ID`                   | none    | Host index within a slice.                           |
| `TPU_NUM_DEVICES`               | none    | TPU devices on the host (if detected).               |
| `TPU_POD_COUNT`                 | none    | Total slices in the pod.                             |
| `TPU_NAME`                      | `EMPTY` | TPU name (if provided by environment).               |
| `TPU_ZONE`                      | `EMPTY` | TPU zone (if provided by environment).               |
| `TPU_VERSION`                   | `v4`    | TPU version for patcher; also propagated to workers. |
| `TPU_SLICE_SIZE`                | `8`     | TPU slice size for patcher.                          |
| `TPU_CORES_PER_HOST`            | `4`     | TPU cores per host for patcher.                      |
| `PATCHER_USER`                  | none    | SSH user for TPU patcher.                            |

## SLURM and CUDA cluster discovery

| Env var                     | Default | What it does                              | Use case                         |
| --------------------------- | ------- | ----------------------------------------- | -------------------------------- |
| `SLURM_JOB_ID`              | none    | Used to choose a coordinator port.        | SLURM job coordination.          |
| `SLURM_STEP_NODELIST`       | none    | Preferred SLURM node list source.         | SLURM cluster discovery.         |
| `SLURM_JOB_NODELIST`        | none    | Fallback SLURM node list source.          | SLURM cluster discovery.         |
| `SLURM_NODELIST`            | none    | Fallback SLURM node list source.          | SLURM cluster discovery.         |
| `SLURM_STEP_TASKS_PER_NODE` | none    | Tasks per node for local process count.   | Correct local device allocation. |
| `SLURMD_NODENAME`           | none    | Current node name in SLURM.               | Map host to node list.           |
| `SLURM_CPUS_ON_NODE`        | none    | CPU count for local scheduling.           | Host CPU resource sizing.        |
| `CUDA_VISIBLE_DEVICES`      | none    | Used to map local device IDs per process. | Device pinning under SLURM.      |

## Notes

- `DEBUG`, `MISSING`, and `MY_VAR` appear only in docstring examples and are not read by the code.
