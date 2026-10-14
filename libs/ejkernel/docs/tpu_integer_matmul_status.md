# TPU integer matmul: API and verification status

Four explicit integer modes and an opt-in streaming A16 backend are
implemented. **This is not a blanket model speedup or quality-parity claim.**
Measurements below were collected on `n_server_spot_m`, TPU v5p, JAX 0.10.0,
2026-09-05. Existing working-tree changes are preserved.

## EasyDeL presets

```python
from easydel.layers.quantization import QuantizationConfig
from easydel.layers.quantization._quants import EasyQuantizer

config = QuantizationConfig.for_matmul("w4a16")
# Other supported presets: w8a16, w4a4, w8a8.
# Optionally restrict conversion to particular model layers:
config.pattern = r".*mlp\.experts\.(gate_up_proj|down_proj)"
quantized_model = EasyQuantizer(quantization_config=config).apply_quantization(model)
```

These presets opt into **explicit** activation precision, including small
batch/decode calls. Legacy configurations retain `activation_policy="auto"`.
A16 means unquantized floating activations, not an implicit conversion of FP32
to BF16. Weight codes are symmetric, per-output-channel signed INT4/INT8;
scales span the contraction dimension. This is not affine INT8 or MXFP4.
Explicit policies reject incompatible storage/runtime formats, simulation,
JAX-native casting, unsupported weight widths, and W8A4 instead of silently
ignoring the requested arithmetic.

## ejkernel operations

For dense operands `[M,K]`, codes `[K,N]`, and scales `[1,N]`:

```python
from ejkernel.modules import channelwise_quantized_matmul

# W4A16 or W8A16, according to codes.dtype:
y = channelwise_quantized_matmul(x, codes, scales, quantize_activations=False)
# W4A4 requires int4 codes; W8A8 uses int8 codes:
y = channelwise_quantized_matmul(
    x, codes, scales, quantize_activations=True,
    activation_bits=4, prefill_threshold=0,
)
```

For routed rows sorted by expert, codes `[E,K,N]`, scales `[E,1,N]`, and
nonnegative int32 group sizes `[E]` summing to at most `M` (remaining rows
are padding with zero output and cotangents):

```python
from ejkernel.modules import grouped_matmul_channelwise

y = grouped_matmul_channelwise(
    x, codes, scales, group_sizes, activation_bits=16,
)
```

Use activation bits 4 or 8 for integer activations. Empty groups are valid.
Per-projection precision is carried through fused MoE dispatch; it must not
be inferred from a single model-wide setting. Standard TP4 and ExpertTensor
forward/JVP/VJP have been tested on TPU. Arbitrary meshes and every model/
backend combination have not been verified.

## Autodiff and numerical contract

Integer weight codes are frozen. Integer-activation paths expose an explicit
straight-through activation derivative using the represented weights, and a
scale derivative using the actual quantized forward activations. This is a
training surrogate, **not** the ordinary derivative of rounding, nor proof
that quantized weight codes can be directly optimized. Dense weight-only
paths retain ordinary autodiff. The shared MLP dispatcher treats packed
quantized fusion as inference-only; training uses the differentiable
composition. Direct AD through that packed inference-only primitive is not
promised.

Do not put integer code arrays in an optimizer. For example, scale-only
fine-tuning can select floating scale parameters explicitly:

```python
import optax
import spectrax as spx
from easydel.infra.base_state import EasyDeLState

state = EasyDeLState.create(
    model=quantized_model,
    trainable_selector=spx.path_endswith("quant_scales"),
    tx=optax.sgd(1e-3), init_opt_state=True,
)
```

A small quantized Llama test verifies a causal-LM loss updates these scales
and leaves frozen state unchanged for all four modes. This is not full-weight
QAT or a claim of training speedup.

FP32 activation/tangent dots request high precision where needed to avoid
silent TPU BF16 rounding. Row quantization corrects exact BF16 half-way values
that TPU reciprocal lowering can otherwise round to the wrong integer.

## Measured kernel results

Host-synchronized medians; compilation excluded. Random BF16 activations and
common random source weights, with independent represented-code checks.
These are single-device grouped operations, **not serving TPS**. The BF16
baseline uses XLA grouped matmul with an explicit valid tile size; further
comparison against tuned existing kernels is still needed.

| `[E,M,K,N]` | BF16 ms | W4A16 ms | W8A16 ms | W4A4 ms | W8A8 ms |
|---|---:|---:|---:|---:|---:|
| `[128,24,2560,1280]` | 1.072 | 0.677 | 0.765 | 0.489 | 0.178 |
| `[128,80,640,2560]` | 1.685 | 0.488 | 0.535 | 0.734 | 0.215 |

W8A8 was fastest on these shapes. W4A4 relative L2 error against BF16 was
approximately 0.20–0.22; W4A16 about 0.14–0.15; eight-bit modes about
0.008–0.012. These synthetic errors do not establish model quality.
Dense small-shape timings clustered around 0.13–0.14 ms, limited by dispatch
and synchronization; no meaningful dense speedup is established there.

The current grouped A16 path has large compiled temporary-buffer estimates:
839,228,736 bytes for the first shape and 420,511,008 for the second. These
are **compiler estimates, not measured peak HBM**. Packed persistent weights
do not guarantee packed intermediates or a bandwidth win.

## MXFP4-to-integer experiment

The prototype preserves packed nibbles and the repository's signed exponent
bytes. This exponent convention is not standard OCP biased E8M0. With both
operands already FP4 and contraction-axis groups of 32, doubled FP4 values
have magnitudes `0,1,2,3,4,6,8,12`:

- They fit INT8 exactly, but not a single signed INT4 value.
- An exact two-plane decomposition `z = low + 2*high` fits each plane in INT4.
- Decomposing both operands requires four integer products per scale group.
- Arbitrary BF16 activations are **not** converted exactly by this trick.

A bounded exponent experiment (`[-3,3]`) compared against an independent
floating reference. For `[M,K,N]=[128,1024,512]`, full decode plus one BF16 dot
measured 0.139 ms, grouped INT8 computation 0.272 ms, and four INT4 plane
products 0.536 ms. Thus **no MXFP4 integer speedup has been demonstrated**.
Signed-zero behavior, extreme exponents, and bitwise IEEE accumulation
identity are not established; represented finite-value equivalence must not
be confused with bitwise equivalence.

## Verification and remaining work

This continuation ran 110 focused CPU tests, then 80 integration/config tests
after stricter validation, and 51 TPU numerical/autodiff/tail/model-block
tests. Some suites overlap; do not sum them as unique coverage. CPU runs set
both `ENABLE_DISTRIBUTED_INIT=0` and `JAX_PLATFORMS=cpu`.

### Further round-2 verification

The legacy full-checkpoint Qwen baseline was rerun: **29.16 decode TPS on each
of eight streams**, 149.3 aggregate end-to-end TPS, 9.86 s mean TTFT, and
44.4 GiB/device. All eight 512-token outputs matched warmup exactly. Context
was configured to 262144 but prompts were 1024 tokens. This is a legacy
baseline, not evidence of speed for the new explicit modes.

The matching **W4A4** run completed at **14.20 decode TPS/stream**, 58.6
aggregate end-to-end TPS, 33.81 s TTFT, and **29.8 GiB/device**. Warmup/timed
outputs again matched exactly (8×512). Across modes, tokens differed on seven
of eight identical prompts. This demonstrates a memory saving and a speed
regression, not equal model quality. No held-out quality evaluation has been
performed. The W4A4 run predates the later AD/padding fixes described below.

The final **explicit W8A8** rerun after those fixes completed at **31.29
 decode TPS/stream**, **155.7 aggregate end-to-end TPS**, 9.92 s mean TTFT,
and **44.4 GiB/device**. This is about 7.3% higher decode throughput than the
legacy baseline on this workload. Every stream generated 512 tokens; all
warmup/timed outputs matched, and all outputs matched the preceding W8A8
run. Outputs differ from legacy on five of eight prompts, so this is not a
bit-exact or quality-equivalent comparison. A held-out quality evaluation is
still needed before adopting it broadly.

Thirty additional CPU tests passed covering DeepSeek split projections with
standard/hash routing, integer MLP training dispatch, routing-permutation
AD, and TP4/ExpertTensor forward/JVP/VJP. The distributed reference explicitly
models shard-local down-activation calibration in standard TP versus
whole-row calibration in ExpertTensor. A4/A8 results are therefore not
expected to match across layouts. CPU ExpertTensor uses ring collectives,
not the TPU ragged-all-to-all implementation.

These tests exposed and fixed two real forward-mode AD defects: the old
custom-VJP-only token permutation, and traced codes/group sizes captured by
the grouped operation's custom-JVP closure. The latter required passing
frozen integer arrays as explicit operands so AD of an already-compiled
call also works. Subsequent TPU tests exposed a separate replicated expert
fan-out transpose defect: overlapping ragged receive writes returned 1/4 of
the expected gradient on four devices. The forward collective is retained;
a gather-plus-psum tangent now gives the required additive transpose.

After correcting the test oracle to request FP32 matmul precision, the final
TPU gate passed **40 tests**, including all twelve TP4/ExpertTensor
forward/JVP/VJP cases using actual TPU all-to-all, nine streaming tests,
DeepSeek routing, MLP training dispatch, and the minimal fan-out reproducer.
No tolerance was relaxed. These tests overlap earlier focused runs.

A streaming A16 prototype using existing Pallas v3 with activation quantization
disabled passes the represented-code reference on tested shapes. Wider
`(M,K,N)=(16,512,1024)` tiles reduced W4A16 time from 0.680 to 0.299 ms for
`[E,M,K,N]=[128,24,2560,1280]`, while compiled temporary estimates fell from
839,152,960 to 172,000 bytes. For `[128,80,640,2560]`, time was 0.490 versus
0.484 ms and temporary estimates 419,755,296 versus 917,408 bytes. These are
prototype results, not a new automatic dispatch path or peak-HBM measurement.
The initial small-tile prototype was substantially slower, so tile choice
matters. A registered **opt-in** public implementation is now available:

```python
y = grouped_matmul_channelwise(
    x_bf16, codes, scales, group_sizes,
    activation_bits=16, platform="pallas",
)
```

This path requires TPU hardware, BF16 activations and signed INT4/INT8 codes.
It supports JVP/VJP through an XLA derivative reference, which may materialize
floating weights: no backward memory saving is claimed. `platform="xla"`
remains the default, including existing model dispatch. The public streaming
path passed hardware primal, cached-JIT AD, tiling and dtype tests; the timing
numbers above were measured on its underlying prototype, not a serving model.

The final public-API benchmark also passed represented-code checks. For
`[128,24,2560,1280]`, W4A16 was 0.684 ms XLA versus 0.305 ms Pallas; W8A16
was 0.772 versus 0.300 ms. Temporary-buffer estimates were 839,326,464 versus
172,000 bytes. For `[128,80,640,2560]`, W4A16 was 0.497 versus 0.489 ms and
W8A16 0.544 versus 0.490 ms, with 420,608,736 versus 917,408 temporary bytes.

## Dense model measurement

A small random-weight Llama (four layers, hidden 512, intermediate 1536,
batch 8×32 tokens) was tested on TPU with model parameters passed as runtime
inputs, not compile-time constants. Host-synchronized median forward times:

| Mode | Milliseconds | Logit relative L2 vs BF16 |
|---|---:|---:|
| BF16 | 0.809 | 0 |
| W4A16 | 0.967 | 0.355 |
| W8A16 | 0.967 | 0.0228 |
| W4A4 | 1.020 | 0.487 |
| W8A8 | 1.015 | 0.0311 |

Quantization did not speed up this small model. These random-weight logit
errors are not model-quality scores. Full-checkpoint Qwen A16 performance
was not measured; its A16 expert correctness and distributed AD were tested.

## MXFP4 activation approximation and scope limits

The runnable experiments are `benchmarks/mxfp4_integer_probe.py` (both operands
already FP4) and `benchmarks/mxfp4_bf16_activations_probe.py` (BF16 activations).
In the latter, FP4 weight storage/decoding stays unchanged, but per-K32-block
INT8 activation quantization is an approximation. At `[128,1024,512]`, it
measured 0.310 ms versus 0.138 ms for decoded BF16 matmul, with relative L2
error 0.00533 against FP4-weight/A16 computation. Again, no integer MXFP4
speedup was demonstrated.

The independent review found no blocking AD issue; its padded-buffer coverage
request exposed a real TPU issue. Explicit masks now prevent uninitialized
ragged-dot tails from contaminating outputs and scale gradients. Sixteen TPU
padding tests passed, including all-empty groups and NaN-poisoned unused rows.

Final acceptance gate: **106 TPU tests passed in 129.40 s**, including actual
scale-only optimizer steps for all four modes. The focused CPU contract suite
passed **142 tests, with 4 TPU-only skips**, and a checkpoint/QAT/STE/layout
compatibility batch passed **101 tests**. Suites overlap and are not the
entire repository test suite. Deprecation warnings were reported for SWIG and
`jax.core.is_concrete`; they were not suppressed. `compileall` and
`git diff --check` passed. Changes remain uncommitted.

GPU execution, arbitrary TPU generations/mesh layouts, zero-sized physical
operand dimensions, extreme FP4 exponents/IEEE bitwise equivalence, direct AD
through inference-only packed MLP primitives, and full-weight integer-code
optimization are not claimed supported or verified here. No default runtime
policy was switched automatically. Quantization quality requires evaluation
on the intended model/task; the synthetic prompt benchmark is not that test.
