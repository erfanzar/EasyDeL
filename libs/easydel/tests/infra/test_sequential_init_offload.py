"""Hardware-only initialization placement regressions; not run on CPU."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.utils import ArrayParam
from easydel.layers.linears import ParallelLinear
from jax import numpy as jnp


def _checked_ones(key, shape, dtype):
    value = jnp.ones(shape, dtype)
    # lazy_init intentionally traces initializers under eval_shape; tracers
    # have no concrete device. Check physical allocations during materialization.
    if not isinstance(key, jax.core.Tracer) and jax.config.jax_default_device is not None:
        device = jax.config.jax_default_device
        # Check the allocation itself, not just its later device_put result.
        assert key.devices() == {device}
        assert value.devices() == {device}
    return value


class _ResureBlock(spx.Module):
    def __init__(self):
        self.table = spx.Parameter(jnp.ones((3,), jnp.float32), sharding=(None,))

    def resure(self, key, shard_fn):
        device = jax.config.jax_default_device
        if device is not None:
            assert key.devices() == {device}
        value = jnp.full((3,), 3.0)
        if device is not None:
            assert value.devices() == {device}
        self.table.value = shard_fn(value)


class _OffloadProbe(EasyDeLBaseModule):
    _model_task = None
    _model_type = "review_init_placement"

    def __init__(self, config, rngs):
        super().__init__(config, jnp.float32, jnp.float32, None, rngs)
        self.blocks = spx.nn.ModuleList([])
        for index in range(2):
            with spx.assign_stage(total=2, current=index):
                self.blocks.append(ParallelLinear(4, 4, kernel_init=_checked_ones, bias_init=_checked_ones, rngs=rngs))
        self.extra = ArrayParam.bound((3,), jnp.float32, "ones")
        self.restored = _ResureBlock()

    def forward(self, inputs):
        return self.blocks[0](inputs)


@pytest.mark.parametrize("pipeline_width", [1, 2])
def test_sequential_init_offload_preserves_training_mesh_and_stage_metadata(pipeline_width):
    if len(jax.devices()) < pipeline_width:
        pytest.skip("Not enough accelerator devices for the requested training pipeline")
    config = EasyDeLBaseConfig(sharding_axis_dims=(pipeline_width, 1, -1, 1, 1, 1))
    training_mesh = config.mesh
    device = jax.local_devices(backend="cpu")[0]
    assert any(training_device.platform != "cpu" for training_device in training_mesh.devices.flat)
    # Supply accelerator RNG state to ensure it is moved before initialization.
    rngs = spx.Rngs(17)
    previous_default_device = jax.config.jax_default_device
    model = _OffloadProbe.sequential_init(config=config, rngs=rngs, init_device=device)

    assert model.config is config
    assert model.mesh is training_mesh
    assert config.mesh is training_mesh
    assert config.sharding_axis_dims == (pipeline_width, 1, -1, 1, 1, 1)
    assert model.blocks[0].weight.stage_assignment == (0, 2)
    assert model.blocks[1].weight.stage_assignment == (1, 2)
    parameters = [var for _, var in spx.iter_variables(model) if var.kind == "parameters"]
    assert parameters
    for var in parameters:
        assert isinstance(var.value, jax.Array)
        assert var.value.devices() == {device}
        assert var.value.dtype == jnp.float32
    for block in model.blocks:
        np.testing.assert_array_equal(np.asarray(block.weight.value), np.ones((4, 4), np.float32))
        np.testing.assert_array_equal(np.asarray(block.bias.value), np.ones((4,), np.float32))
    np.testing.assert_array_equal(np.asarray(model.extra.value), np.ones((3,), np.float32))
    np.testing.assert_array_equal(np.asarray(model.restored.table.value), np.full((3,), 3.0, np.float32))
    assert jax.config.jax_default_device is previous_default_device
