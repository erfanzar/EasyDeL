---
name: port-ejkernel-to-easydel-operation
description: Wire an existing or newly added ejkernel public module operation into EasyDeL as a real OperationImpl adapter under libs/easydel/easydel/operations/kernels. Use when EasyDeL model, training, or inference paths should call ejkernel.modules operations such as paged flash attention, GDN, ragged GDR, ragged conv1d, Qwen3 Next kernels, or eSurge inference kernels. If the kernel core still lives in EasyDeL, use add-ejkernel-kernel first, then use this skill for the EasyDeL adapter.
---

# Skill: Wire eJKernel Operation Into EasyDeL Operation

This skill covers one edge only: **ejkernel public operation to EasyDeL
`OperationImpl` adapter**.

If core algorithm code still lives in `libs/easydel/easydel/operations/kernels`, do not use this skill to move that core
by itself. First use
`add-ejkernel-kernel` to create the ejkernel operation, backend implementations, config, executor, autotune, and tests.
Then return to this skill to wire the EasyDeL runtime to that ejkernel operation.

Load and follow `run-research` first for non-trivial ports. Load
`add-ejkernel-kernel` first only when the required ejkernel operation or backend does not already exist.

## Required Mental Model

Use the repo's words precisely:

- `Platform` is the implementation family: `Platform.XLA`, `Platform.PALLAS`,
  `Platform.TRITON`, `Platform.CUDA`, `Platform.CUTE`, or `Platform.TILELANG`.
- `Backend` is the hardware family: `Backend.CPU`, `Backend.GPU`,
  `Backend.TPU`, or `Backend.ANY`.
- EasyDeL adapters are not kernel implementations. They adapt EasyDeL metadata, caches, dtype, masks, shardings, and
  operation outputs to an ejkernel public module operation.
- XLA code must not call Pallas. Pallas code must not be registered as XLA.
- EasyDeL operation files must not import private ejkernel backend paths such as
  `ejkernel.kernels._pallas`, `ejkernel.kernels._xla`, or
  `ejkernel.kernels._triton`.

If you cannot state the intended EasyDeL operation name, ejkernel public module operation, and backend
`(Platform, Backend)` pairs before editing, stop and read the implementation again.

## First Reads

Before editing, read the concrete source and call graph:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- `libs/ejkernel/pyproject.toml`
- `libs/easydel/easydel/operations/kernels/paged_flash_attention.py`
- the target EasyDeL operation file under
  `libs/easydel/easydel/operations/kernels/`, if it already exists
- nearby EasyDeL operation adapters, such as
  `libs/easydel/easydel/operations/kernels/unified_attention.py`,
  `libs/easydel/easydel/operations/kernels/decode_attention.py`, or another adapter with the same cache/output shape
- all EasyDeL model, runtime, cache, and executor call sites found with `rg`
- the ejkernel public operation wrapper under
  `libs/ejkernel/ejkernel/modules/operations/`
- the ejkernel config and public export that the EasyDeL adapter should call
- focused ejkernel and EasyDeL tests for the affected operation

For target-hardware performance claims, also use `optimize-ejkernel-kernel` and read its profiling/LLO references. CPU
checks are not TPU or GPU performance evidence.

## Correct EasyDeL Adapter Shape

The EasyDeL side must look like a real EasyDeL operation adapter, not a raw function dump and not a private backend
import.

For operation `foo`, the expected shape is:

- EasyDeL adapter:
  `libs/easydel/easydel/operations/kernels/foo.py`
- ejkernel public operation import from one of:
  `ejkernel.modules`, `ejkernel.modules.operations`, or another documented public ejkernel module surface
- `@OperationRegistry.register`
- `class Foo(OperationImpl)`
- `get_impl_name`
- `get_requirements`
- `forward_native` and backend forwards needed by the local operation pattern
- correct EasyDeL output object, such as `AttentionOutput`, or the existing operation contract's return type
- runtime dtype, mesh, sharding, cache, mask, and config taken from
  `self.metadata`, cache views, and operation metadata
- `cfg=self.metadata.get_operation_config("foo")` when the ejkernel operation accepts an operation config
- delegation to the ejkernel public operation for the core computation

Use `paged_flash_attention.py` as the first pattern for cache-backed inference operations:

- import the ejkernel operation from `ejkernel.modules`
- validate EasyDeL cache and metadata before calling the kernel
- derive dtype from `self.metadata.runtime_dtype`
- derive model mode and sharding through EasyDeL metadata helpers
- pass `mesh`, `in_specs`, `out_specs`, and operation config into ejkernel
- return the EasyDeL output type expected by callers
- keep backend methods as thin delegates when the operation pattern does that

## What Is Forbidden

Do not satisfy this skill with a file that only does this:

```python
from ejkernel.modules.operations.foo import foo

__all__ = ("foo",)
```

That may be a compatibility helper module, but it is not an EasyDeL
`OperationImpl` adapter.

Do not put these in an EasyDeL operation adapter:

- `jax.experimental.pallas`
- `pallas_call`
- `pl.`
- imports from `ejkernel.kernels._pallas`
- imports from `ejkernel.kernels._xla`
- imports from `ejkernel.kernels._triton`
- recurrence, convolution, attention, or kernel algorithm bodies that belong in ejkernel
- an XLA implementation that calls Pallas code
- a Pallas implementation registered as `Platform.XLA`

Exception: a small compatibility helper module may re-export a public
`ejkernel.modules.operations` helper when it is not pretending to be an EasyDeL operation. If you use this exception,
name it as a helper, keep it out of the operation registry, and add a structure test proving it does not import private
ejkernel backend paths.

## Port Workflow

1. Inventory before edits:
    - run `git status --short`
    - run `rg -n "<op>|<class>|<helper>" libs/easydel libs/ejkernel`
    - identify whether the ejkernel public operation already exists
2. If the ejkernel operation is missing or incomplete, stop this skill and use
   `add-ejkernel-kernel` first.
3. Write down the exact EasyDeL operation contract:
    - operation name returned by `get_impl_name`
    - input tensors and metadata
    - cache view and cache metadata requirements
    - output type and shape
    - fallback behavior for unsupported backends
4. Implement or repair the EasyDeL adapter as an `OperationImpl`.
5. Update EasyDeL exports, model code, runtime code, or executor code so they call the operation adapter or operation
   registry path, not copied core code.
6. Remove stale EasyDeL core-code bodies only after the adapter has tests.
7. Add or update structure tests and behavioral tests.
8. Run the smallest relevant checks first, then broader EasyDeL and ejkernel checks if call sites changed.

Change one layer at a time. Do not hide a missing ejkernel operation by writing the algorithm directly inside the
EasyDeL adapter.

## Required Structure Checks

Run these checks for the adapter you touched:

```bash
rg -n "ejkernel\.kernels\.|_pallas|_xla|_triton|pallas_call|jax\.experimental\.pallas|pl\\." \
  libs/easydel/easydel/operations/kernels/foo.py
```

This command must have no hits in a real EasyDeL `OperationImpl` adapter.

```bash
rg -n "OperationRegistry\.register|class .*OperationImpl|get_impl_name|get_requirements|forward_native" \
  libs/easydel/easydel/operations/kernels/foo.py
```

This command must show the adapter structure. If it does not, you probably created a helper module, not an EasyDeL
operation.

```bash
rg -n "from ejkernel\.modules|from ejkernel\.modules\.operations|import ejkernel\.modules" \
  libs/easydel/easydel/operations/kernels/foo.py
```

This command should show the public ejkernel module import used by the adapter.

If the EasyDeL file intentionally is only a helper, run the private-import check anyway and explicitly say in the final
report that it is not an
`OperationImpl`.

## Required Tests

Add or update tests that directly validate:

- the EasyDeL operation class is registered by `OperationRegistry`
- `get_impl_name` matches the operation config key used by metadata
- `get_requirements` declares the metadata and cache view required by callers
- the adapter imports ejkernel only through public module surfaces
- representative inputs produce the same result as the previous EasyDeL path or the ejkernel reference operation
- cache-backed operations preserve cache metadata and fallback behavior
- model/runtime call sites use the adapter or registry path after the port

For CPU-safe adapter work, host tests may validate structure, registry wiring, requirements, metadata mapping, and
XLA/reference math.

For TPU Pallas or GPU/Triton behavior, run target-hardware smoke/parity tests. Do not present CPU tests as TPU/GPU
lowering or performance evidence.

Useful focused commands, adjusted to the touched files:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
  uv run pytest libs/easydel/test/<focused-easydel-test>.py
```

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
  uv run pytest libs/ejkernel/test/<focused-ejkernel-test>.py
```

For TPU Pallas behavior:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
  uv run pytest libs/ejkernel/test/kernels/_pallas/tpu/<focused-test>.py
```

## Final Report

Report these items:

- EasyDeL adapter path, class, and `get_impl_name`
- ejkernel public module operation used by the adapter
- metadata, cache, dtype, config, and sharding mapping
- private-kernel import check result
- tests run, with hardware/backend stated plainly
- target-hardware tests or benchmarks skipped, if any, without replacing them with CPU claims
