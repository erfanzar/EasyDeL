"""Hardware-only coverage of freshly initialized Ray trainer offloading."""

import jax
import pytest
import spectrax as spx
from easydel.modules.llama import LlamaConfig, LlamaForCausalLM
from easydel.trainers.ray_scaler.distributed_trainer import RayDistributedTrainer
from jax import numpy as jnp


@pytest.mark.parametrize("lazy", [False, True])
def test_ray_create_model_preserves_training_mesh_and_offloads_parameters(lazy):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    training_mesh = config.mesh
    assert any(device.platform != "cpu" for device in training_mesh.devices.flat)
    launcher = object.__new__(RayDistributedTrainer)
    launcher.offload_backend = "cpu"
    launcher.model_class = LlamaForCausalLM
    previous_default_device = jax.config.jax_default_device
    model = launcher.create_model(config, dtype=jnp.float32, param_dtype=jnp.float32, lazy=lazy)

    assert model.config is config
    assert model.mesh is training_mesh
    assert config.mesh is training_mesh
    assert jax.config.jax_default_device is previous_default_device
    parameters = [var.value for _, var in spx.iter_variables(model) if var.kind == "parameters"]
    assert parameters
    for value in parameters:
        assert value.dtype == jnp.float32
        if lazy:
            assert isinstance(value, jax.ShapeDtypeStruct)
        else:
            assert isinstance(value, jax.Array)
            assert value.devices() == {jax.local_devices(backend="cpu")[0]}
