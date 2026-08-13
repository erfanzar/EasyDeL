# Skills Index

Workflow skills for the EasyDeL monorepo. Start non-trivial work from `run-research`, then layer the domain skill.
Runbooks: `.claude/ops/OPS.md`. See also CLAUDE.md and CLAUDE.md.

- **add-easydel-layer** — Add or update reusable EasyDeL neural-network layers under libs/easydel/easydel/layers. Use
  for attention variants, ParallelLinear, norms, R
- **add-easydel-model** — Add or update an EasyDeL model family, task head, configuration, registry entry, HF conversion
  path, or module test under libs/easydel/easyd
- **add-easydel-operation** — Add or update an EasyDeL native operation or attention kernel under
  libs/easydel/easydel/operations. Use for OperationImpl, OperationRegistr
- **add-easydel-trainer** — Add or update an EasyDeL trainer algorithm under libs/easydel/easydel/trainers. Use when
  creating a new SFT, preference, RL, distillation, r
- **add-easydel-vwhisper** — Add, update, or debug the vWhisper speech-inference engine under
  libs/easydel/easydel/inference/vwhisper. Use for audio transcription/transl
- **add-eformer-optimizer** — Add or update an optimizer or scheduler in libs/eformer/eformer/optimizers. Use for
  OptimizerFactory/SchedulerFactory, new optimizer builder
- **add-ejkernel-kernel** — Add, modify, benchmark, or autotune an ejkernel operation or backend kernel. Use for Pallas
  TPU/GPU, Triton, CUDA, CuTe, TileLang, XLA fallb
- **benchmark-changes** — Benchmark modified EasyDeL components against a baseline — eSurge serving throughput, trainer
  step time, kernel microbenchmarks, packed-pr
- **build-dataset-pipeline** — Build, normalize, pretokenize, pack, mix, save, or validate EasyData datasets for EasyDeL
  training. Use for libs/easydel/easydel/data, Parqu
- **convert-checkpoint** — Convert, verify, download, or publish EasyDeL/Hugging Face checkpoints. Use for
  scripts/convert_hf_to_easydel.py, batch conversion, checkpoi
- **debug-easydel-cache** — Debug or extend EasyDeL inference caches under libs/easydel/easydel/caching. Use for KV
  cache, paged caches, MLA, recurrent, hybrid, turboqu
- **debug-eformer-escale** — Debug or extend mesh and sharding orchestration in libs/eformer/eformer/escale. Use for
  create_mesh, PartitionAxis, PartitionManager, auto_p
- **debug-ejkernel-ops** — Debug or extend the ejkernel execution framework under libs/ejkernel/ejkernel/ops. Use for
  Kernel base class, Executor, ConfigSelectorChain,
- **debug-esurge** — Debug or benchmark EasyDeL eSurge inference. Use for scheduler, runner, executor, KV-cache shape,
  PP/SPMD, DP page placement, no-MTP text se
- **debug-tpu-setup** — Diagnose EasyDeL TPU VM setup, local editable installs from libs, Ray startup, libtpu lock,
  bad-node recovery, or scripts/tpu_setup.sh failu
- **debug-training-oom** — Diagnose EasyDeL training, compile-time HBM OOM, remat/checkpointing, gradient-accumulation,
  chunked loss, or XLA allocator failures. Use fo
- **distributed-review** — Focused review of EasyDeL workspace changes that touch Mesh, NamedSharding, PartitionSpec,
  shard_map, axis policies, FSDP/TP/SP/EP, or pipel
- **docstring-swarm** — Launch a swarm of parallel agents to add and update docstrings across a codebase — module/file,
  class, and function/method docstrings, pub
- **eformer-checkpoint-sharding** — Work on eFormer checkpointing, serialization, fsspec, async checkpoint management,
  mesh creation, partition constraints, or sharding utiliti
- **ejkernel-quantization** — Work on ejkernel weight-compression formats and the quantized_matmul operation under
  libs/ejkernel/ejkernel/quantization and modules/operati
- **generate-tests** — Generate repository-consistent tests for EasyDeL workspace changes — models, trainers,
  operations/kernels, eSurge, data pipelines, infra s
- **optimize-cuda-gpu** — Optimize, profile, or diagnose a CUDA C++ GPU kernel on NVIDIA hardware. Use for CUDA kernel
  performance work — memory coalescing, occupan
- **optimize-ejkernel-kernel** — Optimize, profile, regress, retune, or diagnose an existing ejkernel kernel or
  operation. Use for performance regressions, Pallas TPU/GPU tu
- **optimize-pallas-tpu** — Optimize, profile, or diagnose a JAX Pallas kernel on TPU. Use for Pallas/Mosaic TPU
  performance work — block/tiling choice, VMEM budget,
- **optimize-tilelang-gpu** — Optimize, profile, autotune, or diagnose a TileLang GPU kernel. Use for TileLang
  performance work — tile sizes (block_M/N/K), T.Pipelined
- **optimize-triton-gpu** — Optimize, profile, autotune, or diagnose an OpenAI Triton GPU kernel. Use for Triton
  performance work — block sizes (BLOCK_M/N/K), num_war
- **performance-audit** — Post-implementation performance audit for EasyDeL workspace changes. Use after a feature or
  fix lands to inspect compile cost, memory, commu
- **port-ejkernel-to-easydel-operation** — Wire an existing or newly added ejkernel public module operation into EasyDeL
  as a real OperationImpl adapter under libs/easydel/easydel/ope
- **prepare-commit-pr** — Prepare an EasyDeL commit or pull request safely. Use for staging, pre-commit, import
  layering, focused test selection, PR summaries, branch
- **quantization-layout** — Work on EasyDeL fused projection layout, TP-portable checkpoint layout, quantized linear
  integration, or ejkernel quantized matmul wiring. U
- **release-workspace** — Release or publish one of the EasyDeL uv workspace packages. Use for version bumps, dry-run
  releases, publish tags, standalone repo subtree
- **review-pr** — Multi-agent, high-signal correctness review for EasyDeL pull requests or branch diffs. Use when asked
  to review a PR, review uncommitted cha
- **run-research** — Base workflow for EasyDeL research, optimization, debugging, or multi-step implementation tasks
  that require hypotheses, experiments, eviden
- **spectrax-core** — Work on the SpectraX core object model under libs/spectrax/spectrax/core. Use for Module,
  Variable, Parameter, Buffer, GraphDef, State, Sele
- **spectrax-nn** — Add or update SpectraX neural-network layers and functional primitives under
  libs/spectrax/spectrax/nn and libs/spectrax/spectrax/functional
- **spectrax-pipeline-runtime** — Implement, debug, or benchmark SpectraX module-system and pipeline-runtime behavior.
  Use for pipeline_step, sxcall, sxjit, sxstage_iter, sxs
- **spectrax-sharding** — Work on SpectraX sharding and mesh abstractions under libs/spectrax/spectrax/sharding. Use for
  SpxMesh, create_mesh, PartitionAxis, Partitio
- **spectrax-transforms** — Work on SpectraX module-aware JAX transforms under libs/spectrax/spectrax/transforms. Use
  for spx.jit, grad, vmap, scan, remat, rng_axes, sp
- **test-workspace** — Select and run correct EasyDeL workspace checks. Use for affected-package test planning, CPU JAX
  env setup, import-layering checks, pre-comm
- **tool-reasoning-parser** — Add, debug, or test EasyDeL tool-call and reasoning parsers for OpenAI-compatible
  inference. Use for ToolParserManager, ReasoningParserManag
- **train-elarge** — Create, validate, run, or debug EasyDeL eLarge YAML training, evaluation, or serving configs. Use
  for python -m easydel.scripts.elarge, acti
- **update-docs** — Update EasyDeL workspace documentation when APIs change — regenerate Sphinx api_docs, fix
  hand-written package docs, keep CLAUDE.md/WORKSP
