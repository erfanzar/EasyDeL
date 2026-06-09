# Test Suite and Examples Analysis

## Overview

The ejKernel test suite provides comprehensive testing across all backends and operations, ensuring correctness, performance, and cross-platform compatibility. The test organization follows a clear structure that mirrors the kernel implementations.

## Test Organization

```md
test/
├── run_tests.py              # Main test runner
├── kernels/                  # Kernel-specific tests
│   ├── _xla/                # XLA implementation tests
│   ├── _pallas/             # Pallas implementation tests
│   │   ├── tpu/            # TPU-specific tests
│   │   └── gpu/            # GPU Pallas tests
│   └── _triton/             # Triton GPU tests
├── test_*.py                # Module-level tests
└── benchmarks/              # Performance benchmarks
```

## Main Test Runner

**Location**: `test/run_tests.py`

```python
#!/usr/bin/env python3
"""
Main test runner for ejKernel with platform-specific options.

Usage:
    # Run all tests
    python test/run_tests.py

    # XLA tests only
    python test/run_tests.py --xla

    # Triton tests only
    python test/run_tests.py --triton

    # Comparison tests (XLA vs Triton)
    python test/run_tests.py --comparison

    # Specific pattern
    python test/run_tests.py -k "flash_attention"

    # Verbose with fail-fast
    python test/run_tests.py --verbose --failfast
"""

import unittest
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="ejKernel test runner")

    # Platform selection
    parser.add_argument("--xla", action="store_true",
                       help="Run XLA implementation tests")
    parser.add_argument("--triton", action="store_true",
                       help="Run Triton implementation tests")
    parser.add_argument("--pallas", action="store_true",
                       help="Run Pallas implementation tests")
    parser.add_argument("--comparison", action="store_true",
                       help="Run cross-platform comparison tests")

    # Test options
    parser.add_argument("-k", "--pattern", type=str,
                       help="Run tests matching pattern")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--failfast", "-f", action="store_true",
                       help="Stop on first failure")

    args = parser.parse_args()

    # Determine test directories
    test_dirs = []
    if args.xla:
        test_dirs.append("test/kernels/_xla")
    if args.triton:
        test_dirs.append("test/kernels/_triton")
    if args.pallas:
        test_dirs.append("test/kernels/_pallas")
    if args.comparison:
        test_dirs.append("test/comparison")

    if not test_dirs:
        test_dirs = ["test"]  # All tests

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_dir in test_dirs:
        if args.pattern:
            suite.addTests(loader.discover(test_dir, pattern=f"*{args.pattern}*.py"))
        else:
            suite.addTests(loader.discover(test_dir))

    # Configure runner
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=args.failfast
    )

    # Run tests
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
```

## Test Categories

### 1. Implementation Tests

Test individual kernel implementations for correctness:

```python
# test/kernels/_xla/test_flash_attention.py
import unittest
import jax
import jax.numpy as jnp
from ejkernel.kernels._xla.flash_attention import flash_attention

class TestFlashAttentionXLA(unittest.TestCase):
    """Test XLA Flash Attention implementation"""

    def setUp(self):
        """Setup test fixtures"""
        self.key = jax.random.PRNGKey(42)
        self.batch_size = 2
        self.seq_len = 128
        self.num_heads = 8
        self.head_dim = 64

    def test_basic_attention(self):
        """Test basic attention computation"""
        q, k, v = self._generate_inputs()

        output = flash_attention(
            query=q, key=k, value=v,
            softmax_scale=1.0 / jnp.sqrt(self.head_dim)
        )

        # Check output shape
        self.assertEqual(output.shape, q.shape)

        # Check output is not NaN/Inf
        self.assertFalse(jnp.any(jnp.isnan(output)))
        self.assertFalse(jnp.any(jnp.isinf(output)))

    def test_causal_attention(self):
        """Test causal masking"""
        q, k, v = self._generate_inputs()

        output = flash_attention(
            query=q, key=k, value=v,
            causal=True
        )

        # Verify causal property
        # Attention at position i should not depend on positions > i
        self._verify_causal_property(output)

    def test_attention_with_mask(self):
        """Test with custom attention mask"""
        q, k, v = self._generate_inputs()

        # Create random mask
        mask_key = jax.random.split(self.key)[0]
        mask = jax.random.bernoulli(
            mask_key, p=0.8,
            shape=(self.batch_size, self.seq_len, self.seq_len)
        )

        output = flash_attention(
            query=q, key=k, value=v,
            attention_mask=mask
        )

        # Verify masked positions
        self._verify_masking(output, mask)

    def test_dropout(self):
        """Test attention dropout"""
        q, k, v = self._generate_inputs()

        output1 = flash_attention(
            query=q, key=k, value=v,
            dropout_prob=0.5,
            dropout_seed=123
        )

        output2 = flash_attention(
            query=q, key=k, value=v,
            dropout_prob=0.5,
            dropout_seed=456
        )

        # Different seeds should give different results
        self.assertFalse(jnp.allclose(output1, output2))

        # Same seed should give same results
        output3 = flash_attention(
            query=q, key=k, value=v,
            dropout_prob=0.5,
            dropout_seed=123
        )
        self.assertTrue(jnp.allclose(output1, output3))

    def test_sliding_window(self):
        """Test sliding window attention"""
        q, k, v = self._generate_inputs()

        window_size = 32
        output = flash_attention(
            query=q, key=k, value=v,
            sliding_window=window_size
        )

        # Verify window constraint
        self._verify_sliding_window(output, window_size)

    def test_gradient_computation(self):
        """Test custom VJP implementation"""
        q, k, v = self._generate_inputs()

        def loss_fn(q, k, v):
            output = flash_attention(q, k, v, causal=True)
            return jnp.sum(output)

        # Compute gradients
        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        # Check gradients are valid
        for grad in grads:
            self.assertFalse(jnp.any(jnp.isnan(grad)))
            self.assertFalse(jnp.any(jnp.isinf(grad)))

    def _generate_inputs(self):
        """Generate random Q, K, V tensors"""
        keys = jax.random.split(self.key, 3)

        shape = (self.batch_size, self.seq_len,
                self.num_heads, self.head_dim)

        q = jax.random.normal(keys[0], shape) * 0.02
        k = jax.random.normal(keys[1], shape) * 0.02
        v = jax.random.normal(keys[2], shape) * 0.02

        return q, k, v

    def _verify_causal_property(self, output):
        """Verify causal masking property"""
        # Implementation would check that output at position i
        # doesn't depend on positions > i
        pass

    def _verify_masking(self, output, mask):
        """Verify attention masking"""
        # Implementation would check masked positions
        pass

    def _verify_sliding_window(self, output, window_size):
        """Verify sliding window constraint"""
        # Implementation would check window boundaries
        pass
```

### 2. Comparison Tests

Test consistency across different implementations:

```python
# test/comparison/test_flash_attention_comparison.py
import unittest
import jax.numpy as jnp
from ejkernel import kernel_registry, Platform, Backend

class TestFlashAttentionComparison(unittest.TestCase):
    """Compare Flash Attention across backends"""

    def setUp(self):
        """Get all available implementations"""
        self.implementations = {}

        # Try to get each implementation
        for platform, backend in [
            (Platform.XLA, Backend.ANY),
            (Platform.TRITON, Backend.GPU),
            (Platform.PALLAS, Backend.TPU),
        ]:
            try:
                impl = kernel_registry.get(
                    "flash_attention", platform, backend
                )
                self.implementations[f"{platform}_{backend}"] = impl
            except ValueError:
                # Implementation not available on this system
                pass

        if len(self.implementations) < 2:
            self.skipTest("Need at least 2 implementations for comparison")

    def test_consistency(self):
        """Test all implementations produce consistent results"""
        # Generate inputs
        batch, seq_len, heads, dim = 2, 64, 4, 32
        q = jnp.ones((batch, seq_len, heads, dim))
        k = jnp.ones((batch, seq_len, heads, dim))
        v = jnp.ones((batch, seq_len, heads, dim))

        # Run all implementations
        outputs = {}
        for name, impl in self.implementations.items():
            outputs[name] = impl(
                query=q, key=k, value=v,
                causal=True,
                softmax_scale=0.125
            )

        # Compare all outputs to first
        reference_name = list(outputs.keys())[0]
        reference = outputs[reference_name]

        for name, output in outputs.items():
            if name == reference_name:
                continue

            with self.subTest(comparison=f"{reference_name} vs {name}"):
                # Allow small numerical differences
                jnp.testing.assert_allclose(
                    reference, output,
                    rtol=1e-4, atol=1e-6,
                    err_msg=f"Mismatch between {reference_name} and {name}"
                )

    def test_gradient_consistency(self):
        """Test gradients are consistent across implementations"""
        import jax

        # Create loss function for each implementation
        def make_loss_fn(impl):
            def loss(q, k, v):
                out = impl(q, k, v, causal=True)
                return jnp.sum(out)
            return loss

        # Generate inputs
        q = jnp.ones((1, 32, 2, 16))
        k = jnp.ones((1, 32, 2, 16))
        v = jnp.ones((1, 32, 2, 16))

        # Compute gradients for each implementation
        gradients = {}
        for name, impl in self.implementations.items():
            loss_fn = make_loss_fn(impl)
            grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
            gradients[name] = grads

        # Compare gradients
        reference_name = list(gradients.keys())[0]
        reference = gradients[reference_name]

        for name, grads in gradients.items():
            if name == reference_name:
                continue

            with self.subTest(comparison=f"{reference_name} vs {name}"):
                for i, (ref_grad, grad) in enumerate(zip(reference, grads)):
                    jnp.testing.assert_allclose(
                        ref_grad, grad,
                        rtol=1e-3, atol=1e-5,
                        err_msg=f"Gradient {i} mismatch"
                    )
```

### 3. Performance Tests

Benchmark kernel performance:

```python
# test/benchmarks/test_flash_attention_performance.py
import unittest
import time
import jax
import jax.numpy as jnp
from ejkernel.modules import flash_attention

class TestFlashAttentionPerformance(unittest.TestCase):
    """Performance benchmarks for Flash Attention"""

    def setUp(self):
        """Setup benchmark configuration"""
        self.configs = [
            # (batch, seq_len, heads, head_dim, name)
            (1, 512, 8, 64, "small"),
            (4, 1024, 8, 64, "medium"),
            (8, 2048, 8, 64, "large"),
            (1, 4096, 8, 64, "long_sequence"),
            (32, 512, 8, 64, "large_batch"),
        ]

    def test_forward_performance(self):
        """Benchmark forward pass"""
        results = []

        for batch, seq_len, heads, head_dim, name in self.configs:
            # Generate inputs
            shape = (batch, seq_len, heads, head_dim)
            q = jnp.ones(shape, dtype=jnp.float16)
            k = jnp.ones(shape, dtype=jnp.float16)
            v = jnp.ones(shape, dtype=jnp.float16)

            # Compile
            fn = jax.jit(lambda q, k, v: flash_attention(
                q, k, v, causal=True
            ))

            # Warmup
            for _ in range(3):
                _ = fn(q, k, v).block_until_ready()

            # Benchmark
            start = time.perf_counter()
            iterations = 10
            for _ in range(iterations):
                _ = fn(q, k, v).block_until_ready()
            elapsed = time.perf_counter() - start

            # Calculate metrics
            avg_time = elapsed / iterations
            throughput = (batch * seq_len * seq_len * heads) / avg_time / 1e9

            results.append({
                "config": name,
                "shape": shape,
                "time_ms": avg_time * 1000,
                "throughput_gflops": throughput
            })

            print(f"{name}: {avg_time*1000:.2f}ms, "
                  f"{throughput:.1f} GFLOPS")

        # Store results for analysis
        self._results = results

    def test_memory_usage(self):
        """Test memory consumption"""
        import jax.profiler

        for batch, seq_len, heads, head_dim, name in self.configs[:3]:
            with self.subTest(config=name):
                shape = (batch, seq_len, heads, head_dim)
                q = jnp.ones(shape, dtype=jnp.float16)
                k = jnp.ones(shape, dtype=jnp.float16)
                v = jnp.ones(shape, dtype=jnp.float16)

                # Get memory before
                device = jax.devices()[0]
                if hasattr(device, "memory_stats"):
                    mem_before = device.memory_stats()["bytes_in_use"]
                else:
                    self.skipTest("Memory profiling not available")

                # Run kernel
                output = flash_attention(q, k, v, causal=True)
                output.block_until_ready()

                # Get memory after
                mem_after = device.memory_stats()["bytes_in_use"]
                mem_used = mem_after - mem_before

                # Calculate theoretical minimum
                # O(N) memory for Flash Attention
                theoretical = batch * seq_len * heads * head_dim * 2  # FP16

                print(f"{name}: Used {mem_used/1e6:.1f}MB, "
                      f"Theoretical: {theoretical/1e6:.1f}MB")

                # Flash Attention should use much less than O(N^2)
                quadratic = batch * seq_len * seq_len * heads * 2
                self.assertLess(mem_used, quadratic * 0.1,
                              "Memory usage too high for Flash Attention")
```

### 4. Module Tests

Test high-level module interfaces:

```python
# test/test_module_flash_attention.py
import unittest
from unittest.mock import Mock, patch
import jax.numpy as jnp
from ejkernel.modules.operations import FlashAttention, FlashAttentionConfig
from ejkernel.ops.utils.datacarrier import FwdParams, BwdParams

class TestFlashAttentionModule(unittest.TestCase):
    """Test Flash Attention module interface"""

    def test_configuration_handling(self):
        """Test configuration creation and processing"""
        module = FlashAttention()

        # Test with dict input
        config = FlashAttentionConfig(
            fwd_params={"q_blocksize": 128, "kv_blocksize": 256},
            bwd_params={"q_blocksize": 64, "kv_blocksize": 128}
        )

        # Should auto-convert to dataclasses
        self.assertIsInstance(config.fwd_params, FwdParams)
        self.assertIsInstance(config.bwd_params, BwdParams)
        self.assertEqual(config.fwd_params.q_blocksize, 128)

    def test_heuristic_generation(self):
        """Test default configuration generation"""
        module = FlashAttention()

        # Create mock invocation
        inv = Mock()
        inv.kwargs = {
            "query": jnp.ones((2, 128, 8, 64)),
            "key": jnp.ones((2, 128, 8, 64)),
            "value": jnp.ones((2, 128, 8, 64)),
        }

        config = module.heuristic_cfg(inv)

        self.assertIsNotNone(config.fwd_params)
        self.assertIsNotNone(config.bwd_params)
        self.assertEqual(config.platform, "auto")

    def test_candidate_generation(self):
        """Test autotuning candidate generation"""
        module = FlashAttention()

        inv = Mock()
        inv.kwargs = {
            "query": jnp.ones((2, 512, 8, 128)),
            "key": jnp.ones((2, 512, 8, 128)),
            "value": jnp.ones((2, 512, 8, 128)),
            "sliding_window": None
        }

        # Test GPU candidates
        with patch("ejkernel.modules.operations.flash_attention.get_device_platform",
                  return_value="gpu"):
            candidates = module.candidate_cfgs(inv)

            self.assertGreater(len(candidates), 0)
            for cfg in candidates:
                self.assertIsInstance(cfg, FlashAttentionConfig)
                self.assertIsNotNone(cfg.fwd_params.q_blocksize)
                self.assertIsNotNone(cfg.fwd_params.kv_blocksize)

    def test_distributed_wrapper(self):
        """Test shard_map wrapper creation"""
        module = FlashAttention()

        q = jnp.ones((2, 128, 8, 64))
        k = jnp.ones((2, 128, 8, 64))
        v = jnp.ones((2, 128, 8, 64))

        mesh = Mock()
        in_specs = (None, None, None)
        out_specs = None

        # Test wrapper creation
        wrapper_fn, args, kwargs = module.create_shard_map_wrapper(
            q, k, v,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            causal=True
        )

        self.assertIsNotNone(wrapper_fn)
        self.assertEqual(len(args), 3)
        self.assertTrue(kwargs.get("causal"))
```

### 5. Integration Tests

End-to-end tests with real models:

```python
# test/integration/test_transformer_block.py
import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
from ejkernel.modules import flash_attention

class TransformerBlock(nn.Module):
    """Simple transformer block using ejKernel"""
    num_heads: int = 8
    head_dim: int = 64
    mlp_dim: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, training=False):
        # Self-attention with Flash Attention
        attn_output = self.attention(x, mask, training)

        # Add & Norm
        x = nn.LayerNorm()(x + attn_output)

        # FFN
        ffn_output = self.ffn(x, training)

        # Add & Norm
        x = nn.LayerNorm()(x + ffn_output)

        return x

    def attention(self, x, mask, training):
        batch, seq_len, dim = x.shape

        # Linear projections
        q = nn.Dense(self.num_heads * self.head_dim)(x)
        k = nn.Dense(self.num_heads * self.head_dim)(x)
        v = nn.Dense(self.num_heads * self.head_dim)(x)

        # Reshape for attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Apply Flash Attention
        attn_output = flash_attention(
            q, k, v,
            attention_mask=mask,
            causal=True,
            dropout_prob=self.dropout_rate if training else 0.0
        )

        # Reshape and project
        attn_output = attn_output.reshape(batch, seq_len, -1)
        return nn.Dense(dim)(attn_output)

    def ffn(self, x, training):
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(x.shape[-1])(x)
        return x


class TestTransformerIntegration(unittest.TestCase):
    """Test ejKernel integration with transformer models"""

    def test_transformer_forward(self):
        """Test forward pass through transformer block"""
        model = TransformerBlock(num_heads=4, head_dim=32)

        # Initialize
        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 64, 128))
        params = model.init(key, x)

        # Forward pass
        output = model.apply(params, x)

        self.assertEqual(output.shape, x.shape)
        self.assertFalse(jnp.any(jnp.isnan(output)))

    def test_transformer_gradient(self):
        """Test gradient computation through transformer"""
        model = TransformerBlock(num_heads=4, head_dim=32)

        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 64, 128))
        params = model.init(key, x)

        def loss_fn(params, x):
            output = model.apply(params, x)
            return jnp.mean(output ** 2)

        # Compute gradients
        grads = jax.grad(loss_fn)(params, x)

        # Check gradients are valid
        for layer_grads in jax.tree_leaves(grads):
            self.assertFalse(jnp.any(jnp.isnan(layer_grads)))
```

## Example Usage Scripts

### Basic Attention Example

```python
# examples/basic_attention.py
"""
Basic example of using Flash Attention from ejKernel.
"""

import jax
import jax.numpy as jnp
from ejkernel.modules import flash_attention

def main():
    # Setup
    key = jax.random.PRNGKey(42)
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64

    # Generate random inputs
    shape = (batch_size, seq_len, num_heads, head_dim)
    q = jax.random.normal(key, shape) * 0.02
    k = jax.random.normal(jax.random.split(key)[0], shape) * 0.02
    v = jax.random.normal(jax.random.split(key)[1], shape) * 0.02

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")

    # Basic attention
    output = flash_attention(q, k, v)
    print(f"Basic attention output: {output.shape}")

    # Causal attention
    causal_output = flash_attention(q, k, v, causal=True)
    print(f"Causal attention output: {causal_output.shape}")

    # With dropout
    dropout_output = flash_attention(
        q, k, v,
        causal=True,
        dropout_prob=0.1,
        dropout_seed=123
    )
    print(f"Attention with dropout: {dropout_output.shape}")

    # With sliding window
    window_output = flash_attention(
        q, k, v,
        causal=True,
        sliding_window=32
    )
    print(f"Sliding window attention: {window_output.shape}")

    # With soft capping (Gemma-2 style)
    capped_output = flash_attention(
        q, k, v,
        causal=True,
        logits_soft_cap=30.0
    )
    print(f"Soft-capped attention: {capped_output.shape}")

if __name__ == "__main__":
    main()
```

### Configuration Override Example

```python
# examples/custom_config.py
"""
Example of using custom configuration with ejKernel.
"""

from ejkernel.modules import flash_attention, FlashAttentionConfig
from ejkernel.ops.utils.datacarrier import FwdParams, BwdParams
import jax.numpy as jnp

def main():
    # Create custom configuration
    custom_config = FlashAttentionConfig(
        fwd_params=FwdParams(
            q_blocksize=256,    # Larger blocks for forward
            kv_blocksize=256,
            num_warps=8,
            num_stages=2
        ),
        bwd_params=BwdParams(
            q_blocksize=128,    # Smaller blocks for backward
            kv_blocksize=128,
            num_warps=4,
            num_stages=1
        ),
        platform="triton",      # Force Triton implementation
        backend="gpu"
    )

    # Generate inputs
    q = jnp.ones((2, 512, 8, 64))
    k = jnp.ones((2, 512, 8, 64))
    v = jnp.ones((2, 512, 8, 64))

    # Use custom configuration
    output = flash_attention(
        q, k, v,
        causal=True,
        cfg=custom_config
    )

    print(f"Output with custom config: {output.shape}")

if __name__ == "__main__":
    main()
```

### Distributed Attention Example

```python
# examples/distributed_attention.py
"""
Example of distributed Flash Attention with shard_map.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from ejkernel.modules import flash_attention

def main():
    # Setup devices
    devices = jax.devices()
    print(f"Available devices: {devices}")

    # Create mesh
    mesh = Mesh(devices, axis_names=("data",))

    # Generate sharded inputs
    shape = (8, 1024, 8, 64)  # Large batch for distribution
    key = jax.random.PRNGKey(0)

    with mesh:
        # Create inputs with sharding
        q = jax.random.normal(key, shape)
        k = jax.random.normal(jax.random.split(key)[0], shape)
        v = jax.random.normal(jax.random.split(key)[1], shape)

        # Shard across batch dimension
        q = jax.device_put(q, jax.sharding.NamedSharding(
            mesh, P("data", None, None, None)
        ))
        k = jax.device_put(k, jax.sharding.NamedSharding(
            mesh, P("data", None, None, None)
        ))
        v = jax.device_put(v, jax.sharding.NamedSharding(
            mesh, P("data", None, None, None)
        ))

        # Run distributed attention
        output = flash_attention(
            q, k, v,
            causal=True,
            mesh=mesh,
            in_specs=(P("data", None, None, None),) * 3,
            out_specs=P("data", None, None, None)
        )

        print(f"Distributed output shape: {output.shape}")
        print(f"Output sharding: {output.sharding}")

if __name__ == "__main__":
    main()
```

## Test Utilities

### Test Base Class

```python
# test/test_utils.py
import unittest
import jax
import jax.numpy as jnp

class KernelTestCase(unittest.TestCase):
    """Base class for kernel tests with utilities"""

    def assert_allclose(self, actual, expected, rtol=1e-5, atol=1e-8):
        """Assert arrays are close with better error messages"""
        try:
            jnp.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as e:
            # Add more context
            diff = jnp.abs(actual - expected)
            max_diff = jnp.max(diff)
            max_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)

            raise AssertionError(
                f"{str(e)}\n"
                f"Max difference: {max_diff} at index {max_idx}\n"
                f"Actual value: {actual[max_idx]}\n"
                f"Expected value: {expected[max_idx]}"
            )

    def assert_no_nan_inf(self, array, name="array"):
        """Assert array contains no NaN or Inf values"""
        self.assertFalse(
            jnp.any(jnp.isnan(array)),
            f"{name} contains NaN values"
        )
        self.assertFalse(
            jnp.any(jnp.isinf(array)),
            f"{name} contains Inf values"
        )

    def generate_attention_inputs(self, batch=2, seq_len=128,
                                 heads=8, head_dim=64, dtype=jnp.float32):
        """Generate standard attention inputs"""
        key = getattr(self, 'key', jax.random.PRNGKey(0))
        keys = jax.random.split(key, 3)

        shape = (batch, seq_len, heads, head_dim)

        q = jax.random.normal(keys[0], shape, dtype=dtype) * 0.02
        k = jax.random.normal(keys[1], shape, dtype=dtype) * 0.02
        v = jax.random.normal(keys[2], shape, dtype=dtype) * 0.02

        return q, k, v
```

## Test Coverage

The test suite aims for comprehensive coverage:

1. **Functional Correctness**: All operations produce correct results
2. **Numerical Stability**: No NaN/Inf in outputs or gradients
3. **Cross-platform Consistency**: Same results across backends
4. **Performance Regression**: Track performance over time
5. **Integration**: Works correctly in real models
6. **Edge Cases**: Handle empty inputs, extreme sizes, etc.

## Continuous Integration

The project can be integrated with CI/CD:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
        backend: [cpu, gpu]

    steps:
    - uses: actions/checkout@v2

    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e .[dev]
        if [ "${{ matrix.backend }}" == "gpu" ]; then
          pip install jax[cuda12]
        fi

    - name: Run tests
      run: |
        if [ "${{ matrix.backend }}" == "gpu" ]; then
          python test/run_tests.py --triton --verbose
        else
          python test/run_tests.py --xla --verbose
        fi

    - name: Run benchmarks
      if: matrix.backend == 'gpu'
      run: |
        python test/benchmarks/run_benchmarks.py
```

## Conclusion

The ejKernel test suite provides:

1. **Comprehensive Coverage**: Tests for all kernels and platforms
2. **Cross-platform Validation**: Ensures consistency across backends
3. **Performance Tracking**: Benchmarks for regression detection
4. **Integration Testing**: Validates real-world usage
5. **Developer Tools**: Utilities for easy test writing

The test organization mirrors the project structure, making it easy to locate and run specific tests while maintaining comprehensive coverage of the entire system.
