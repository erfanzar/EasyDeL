# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Test platform-specific kernel functionality."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
    forward_autotune_only,
    get_device_platform,
)


@dataclass
class TestConfig:
    """Test configuration with different parameters per platform."""

    __test__ = False

    algorithm: str = "default"
    block_size: int = 128


class PlatformSpecificKernel(Kernel[TestConfig, jax.Array]):
    """Example kernel with platform-specific implementations."""

    def __init__(self):
        super().__init__("test_platform_kernel")

    def run(self, x, y, cfg: TestConfig) -> jax.Array:
        """Generic implementation - simple addition."""
        return x + y + jnp.array(cfg.block_size, dtype=x.dtype)

    def heuristic_cfg(self, inv) -> TestConfig:
        """Generic heuristic configuration."""
        return TestConfig(algorithm="generic", block_size=128)

    def candidate_cfgs(self, inv):
        """Generic candidate configurations."""
        return [
            TestConfig(algorithm="generic", block_size=64),
            TestConfig(algorithm="generic", block_size=128),
        ]

    def run_gpu(self, x, y, cfg: TestConfig) -> jax.Array:
        """GPU-optimized implementation - uses multiplication."""
        return x * y * jnp.array(cfg.block_size, dtype=x.dtype)

    def heuristic_cfg_gpu(self, inv) -> TestConfig:
        """GPU-specific heuristic configuration."""
        return TestConfig(algorithm="gpu_optimized", block_size=256)

    def candidate_cfgs_gpu(self, inv):
        """GPU-specific candidate configurations."""
        return [
            TestConfig(algorithm="gpu_fast", block_size=128),
            TestConfig(algorithm="gpu_optimized", block_size=256),
            TestConfig(algorithm="gpu_precise", block_size=512),
        ]

    def run_tpu(self, x, y, cfg: TestConfig) -> jax.Array:
        """TPU-optimized implementation - uses subtraction."""
        return x - y + jnp.array(cfg.block_size, dtype=x.dtype)

    def heuristic_cfg_tpu(self, inv) -> TestConfig:
        """TPU-specific heuristic configuration."""
        return TestConfig(algorithm="tpu_optimized", block_size=1024)

    def candidate_cfgs_tpu(self, inv):
        """TPU-specific candidate configurations."""
        return [
            TestConfig(algorithm="tpu_fast", block_size=512),
            TestConfig(algorithm="tpu_optimized", block_size=1024),
            TestConfig(algorithm="tpu_precise", block_size=2048),
        ]

    def run_cpu(self, x, y, cfg: TestConfig) -> jax.Array:
        """CPU-optimized implementation - uses division."""
        return (x + y) / jnp.array(max(1, cfg.block_size), dtype=x.dtype)

    def heuristic_cfg_cpu(self, inv) -> TestConfig:
        """CPU-specific heuristic configuration."""
        return TestConfig(algorithm="cpu_optimized", block_size=64)

    def candidate_cfgs_cpu(self, inv):
        """CPU-specific candidate configurations."""
        return [
            TestConfig(algorithm="cpu_fast", block_size=32),
            TestConfig(algorithm="cpu_optimized", block_size=64),
        ]


class PlatformSpecificVJPKernel(Kernel[TestConfig, jax.Array]):
    """Example kernel with platform-specific VJP implementations."""

    def __init__(self):
        super().__init__("test_platform_vjp_kernel")

    def run(self, x, y, cfg: TestConfig) -> jax.Array:
        return x @ y

    def heuristic_cfg(self, inv) -> TestConfig:
        return TestConfig(algorithm="generic")

    def fwd_with_residuals(self, x, y, cfg: TestConfig):
        result = x @ y
        return result, (x, y)

    def vjp(self, residuals, output, dy, *args, cfg: TestConfig, **kwargs):
        x, y = residuals
        dx = dy @ y.T
        dy_val = x.T @ dy
        return dx, dy_val

    def fwd_with_residuals_gpu(self, x, y, cfg: TestConfig):
        """GPU-optimized forward pass - potentially with different memory layout."""
        result = jnp.dot(x, y, precision=jax.lax.Precision.HIGH)
        return result, (x, y)

    def vjp_gpu(self, residuals, output, dy, *args, cfg: TestConfig, **kwargs):
        """GPU-optimized backward pass."""
        x, y = residuals
        dx = jnp.dot(dy, y.T, precision=jax.lax.Precision.HIGH)
        dy_val = jnp.dot(x.T, dy, precision=jax.lax.Precision.HIGH)
        return dx, dy_val


class CaptureValidateBackwardTuner(Tuner[TestConfig]):
    """Tuner stub that records whether backward validation was requested."""

    def __init__(self):
        super().__init__(warmup=0, iters=1)
        self.last_validate_backward: bool | None = None

    def autotune(self, make_fn, args, kwargs, candidates):
        candidates = tuple(candidates)
        assert candidates
        fn = make_fn(candidates[0])
        self.last_validate_backward = bool(getattr(fn, "_ejk_validate_backward", False))
        return candidates[0]


class TestPlatformSpecificKernels:
    """Test platform-specific kernel functionality."""

    def test_platform_detection(self):
        """Test that platform detection works correctly."""
        platform = get_device_platform()
        assert platform in ["gpu", "tpu", "cpu", "unknown"]

    def test_platform_specific_run(self):
        """Test that platform-specific run methods are selected."""
        cache = ConfigCache()
        selector = ConfigSelectorChain(cache)
        executor = Executor(selector)

        kernel = PlatformSpecificKernel()
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        result = executor(kernel, x, y)

        assert result.shape == x.shape

        assert jnp.all(jnp.isfinite(result))

    def test_platform_specific_candidate_cfgs(self):
        """Test that platform-specific candidate configs are used."""
        kernel = PlatformSpecificKernel()
        platform = get_device_platform()

        from ejkernel.ops import Invocation

        inv = Invocation(
            op_id=kernel.op_id,
            args=(jnp.array([1.0]), jnp.array([2.0])),
            kwargs={},
        )

        from ejkernel.ops.core import _get_platform_method

        candidate_method = _get_platform_method(kernel, "candidate_cfgs", platform)

        if candidate_method:
            configs = list(candidate_method(inv))

            if platform == "gpu":
                assert any("gpu" in cfg.algorithm for cfg in configs)
            elif platform == "tpu":
                assert any("tpu" in cfg.algorithm for cfg in configs)
            elif platform == "cpu":
                assert any("cpu" in cfg.algorithm for cfg in configs)
        else:
            configs = list(kernel.candidate_cfgs(inv))
            assert any("generic" in cfg.algorithm for cfg in configs)

    def test_platform_specific_heuristic(self):
        """Test that platform-specific heuristic configs are used."""
        cache = ConfigCache()
        selector = ConfigSelectorChain(cache)
        executor = Executor(selector)

        kernel = PlatformSpecificKernel()
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        cfg = executor.choose_config(kernel, x, y)

        platform = get_device_platform()
        if platform == "gpu":
            assert "gpu" in cfg.algorithm
        elif platform == "tpu":
            assert "tpu" in cfg.algorithm
        elif platform == "cpu":
            assert "cpu" in cfg.algorithm
        else:
            assert cfg.algorithm == "generic"

    def test_platform_specific_vjp(self):
        """Test that platform-specific VJP methods work correctly."""
        cache = ConfigCache()
        selector = ConfigSelectorChain(cache)
        executor = Executor(selector)

        kernel = PlatformSpecificVJPKernel()
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = executor(kernel, x, y)
        assert result.shape == (2, 2)

        def f(x_val, y_val):
            return jnp.sum(executor(kernel, x_val, y_val))

        grad_fn = jax.grad(f, argnums=(0, 1))
        dx, dy = grad_fn(x, y)

        assert dx.shape == x.shape
        assert dy.shape == y.shape

        assert jnp.all(jnp.isfinite(dx))
        assert jnp.all(jnp.isfinite(dy))

    def test_forward_autotune_only_disables_backward_autotune_validation(self):
        """forward_autotune_only() should force forward-only autotune timing."""
        kernel = PlatformSpecificKernel()
        inv = Invocation(
            op_id=kernel.op_id,
            args=(jnp.array([1.0]), jnp.array([2.0])),
            kwargs={},
        )

        tuner_default = CaptureValidateBackwardTuner()
        selector_default = ConfigSelectorChain(
            cache=ConfigCache(),
            policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="autotune", validate_backward=True),
            tuner=tuner_default,
        )
        selector_default.choose(inv, kernel)
        assert tuner_default.last_validate_backward is True

        tuner_context = CaptureValidateBackwardTuner()
        selector_context = ConfigSelectorChain(
            cache=ConfigCache(),
            policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="autotune", validate_backward=True),
            tuner=tuner_context,
        )
        with forward_autotune_only():
            selector_context.choose(inv, kernel)
        assert tuner_context.last_validate_backward is False

        tuner_after = CaptureValidateBackwardTuner()
        selector_after = ConfigSelectorChain(
            cache=ConfigCache(),
            policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="autotune", validate_backward=True),
            tuner=tuner_after,
        )
        selector_after.choose(inv, kernel)
        assert tuner_after.last_validate_backward is True

    def test_tuner_measure_backward_accepts_tuple_outputs(self):
        """Backward autotune validation should handle tuple/pytree outputs."""
        tuner = Tuner(warmup=0, iters=1)
        x = jnp.arange(8, dtype=jnp.float32)

        def fn(x_):
            y = x_ * 2.0
            return y, {"mean": jnp.mean(y)}

        fn._ejk_validate_backward = True
        fn._ejk_method = "regular"

        elapsed = tuner.measure(fn, x)
        assert elapsed >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
