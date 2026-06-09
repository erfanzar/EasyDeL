"""
Example: Training with Multiple Quantization Types

This example demonstrates how to use eformer's unified quantization interface
to train models with different quantization formats (NF4, INT8, Binary, etc.).
"""

from functools import partial

import jax
import jax.numpy as jnp

from eformer.jaximus import implicit
from eformer.ops.quantization import QuantizationConfig, QuantizationType, quantize, straight_through


# Example 1: Basic usage with different quantization types
def example_basic_quantization():
    """Show how to quantize with different formats."""
    print("=" * 60)
    print("Example 1: Basic Quantization")
    print("=" * 60)

    weight = jax.random.normal(jax.random.PRNGKey(0), (128, 64), dtype=jnp.float32)

    # NF4 quantization (4-bit, Gaussian-optimized)
    nf4_config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
    nf4_weight = quantize(weight, config=nf4_config)
    print(f"Original weight shape: {weight.shape}, size: {weight.nbytes / 1024:.2f} KB")
    print(f"NF4 packed shape: {nf4_weight.packed.shape}, size: {nf4_weight.packed.nbytes / 1024:.2f} KB")
    print(f"Memory savings: {weight.nbytes / (nf4_weight.packed.nbytes + nf4_weight.absmax.nbytes):.1f}x\n")

    # INT8 quantization
    int8_config = QuantizationConfig(dtype=QuantizationType.INT8, block_size=64)
    int8_weight = quantize(weight, config=int8_config)
    print(f"INT8 shape: {int8_weight.weight.shape}, size: {int8_weight.weight.nbytes / 1024:.2f} KB")
    print(f"Memory savings: {weight.nbytes / (int8_weight.weight.nbytes + int8_weight.scale.nbytes):.1f}x\n")

    # Binary quantization (1-bit)
    binary_weight = quantize(weight, dtype=QuantizationType.BINARY)
    print(f"Binary packed shape: {binary_weight.weight.shape}")
    print(f"Memory savings: ~{weight.nbytes / binary_weight.weight.nbytes:.1f}x\n")


# Example 2: Training with Straight-Through Estimation
def example_training_with_ste():
    """Show training with quantized weights using STE."""
    print("=" * 60)
    print("Example 2: Training with STE")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 32, 128, 64

    # Create data
    inputs = jax.random.normal(key, (batch_size, in_dim), dtype=jnp.float32)
    targets = jax.random.normal(key, (batch_size, out_dim), dtype=jnp.float32)

    # Initialize float32 master weights
    weight_fp32 = jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * 0.01

    # Define loss function with quantization
    @implicit
    def nf4_linear(x, w):
        return x @ w

    def loss_fn(weight, inputs, targets, quant_config):
        # Apply STE: forward uses quantized, backward flows to fp32
        quant_weight = straight_through(weight, config=quant_config)
        preds = nf4_linear(inputs, quant_weight)
        return jnp.mean((preds - targets) ** 2)

    # Train with different quantization types
    configs = [
        ("NF4", QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)),
        ("INT8", QuantizationConfig(dtype=QuantizationType.INT8, block_size=64)),
        ("Binary", QuantizationConfig(dtype=QuantizationType.BINARY)),
    ]

    for name, cfg in configs:
        # JIT compile for performance
        train_step = jax.jit(jax.value_and_grad(partial(loss_fn, quant_config=cfg)))

        # Run a few training steps
        w = weight_fp32.copy()
        print(f"\nTraining with {name}:")
        for step in range(5):
            loss, grad = train_step(w, inputs, targets)
            w = w - 0.01 * grad  # Simple SGD update
            print(f"  Step {step}: loss = {loss:.6f}")


# Example 3: Mixed-precision training
def example_mixed_precision():
    """Show using different quantization for different layers."""
    print("\n" + "=" * 60)
    print("Example 3: Mixed-Precision Quantization")
    print("=" * 60)

    key = jax.random.PRNGKey(0)

    # Define a simple 2-layer network with different quantization per layer
    class MixedQuantModel:
        def __init__(self):
            # Layer 1: NF4 (most aggressive)
            self.w1 = jax.random.normal(key, (128, 256), dtype=jnp.float32) * 0.01
            self.config1 = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)

            # Layer 2: INT8 (less aggressive)
            self.w2 = jax.random.normal(key, (256, 64), dtype=jnp.float32) * 0.01
            self.config2 = QuantizationConfig(dtype=QuantizationType.INT8, block_size=64)

        def forward(self, x):
            # Quantize each layer with its own config
            w1_quant = straight_through(self.w1, config=self.config1)
            h = jax.nn.relu(x @ w1_quant)

            w2_quant = straight_through(self.w2, config=self.config2)
            out = h @ w2_quant
            return out

    model = MixedQuantModel()
    x = jax.random.normal(key, (32, 128), dtype=jnp.float32)
    output = model.forward(x)
    print(f"Output shape: {output.shape}")
    print("Layer 1 uses NF4 (4-bit), Layer 2 uses INT8 (8-bit)")


# Example 4: Simulation mode for QAT
def example_simulation_mode():
    """Show using simulation mode for quantization-aware training."""
    print("\n" + "=" * 60)
    print("Example 4: Simulation Mode (QAT)")
    print("=" * 60)

    weight = jax.random.normal(jax.random.PRNGKey(0), (128, 64), dtype=jnp.float32)

    # Normal mode: actual bit packing
    config_packed = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64, simulate=False)
    packed = quantize(weight, config=config_packed)
    print(f"Packed mode: returns {type(packed).__name__}")
    print(f"  - Packed data shape: {packed.packed.shape}")
    print("  - Memory saved: ~8x")

    # Simulation mode: no bit packing, just simulates quantization
    config_sim = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64, simulate=True)
    simulated = quantize(weight, config=config_sim)
    print(f"\nSimulation mode: returns {type(simulated).__name__}")
    print(f"  - Shape: {simulated.shape}")
    print(f"  - Dtype: {simulated.dtype}")
    print("  - No memory savings, but faster for QAT experiments")


# Example 5: Inference with quantized weights
def example_inference():
    """Show using quantized weights for inference."""
    print("\n" + "=" * 60)
    print("Example 5: Inference with Quantized Weights")
    print("=" * 60)

    key = jax.random.PRNGKey(0)
    weight_fp32 = jax.random.normal(key, (128, 64), dtype=jnp.float32)

    # After training, quantize for inference
    nf4_weight = quantize(weight_fp32, dtype=QuantizationType.NF4, block_size=64)

    @implicit
    @jax.jit
    def inference(x, w):
        return x @ w  # Uses NF4 TPU kernel automatically

    # Run inference
    inputs = jax.random.normal(key, (32, 128), dtype=jnp.float32)
    outputs = inference(inputs, nf4_weight)

    print(f"Input shape: {inputs.shape}")
    print(f"Weight stored as NF4: {nf4_weight.packed.shape}")
    print(f"Output shape: {outputs.shape}")
    print("TPU kernel used automatically (if available)")


if __name__ == "__main__":
    example_basic_quantization()
    example_training_with_ste()
    example_mixed_precision()
    example_simulation_mode()
    example_inference()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
