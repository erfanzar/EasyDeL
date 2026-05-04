import jax
import jax.numpy as jnp
import numpy as np

from easydel.inference.esurge.runners.async_types import DeviceInputTokenHandoff
from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from easydel.inference.esurge.runners.execution_types import BatchMetadata


def _metadata(input_ids: jax.Array) -> BatchMetadata:
    return BatchMetadata(
        packed_qsl_seqlens=jnp.zeros((2, 5), dtype=jnp.int32),
        packed_i32_padded=jnp.zeros((3, 4), dtype=jnp.int32),
        packed_f32_padded=jnp.zeros((6, 4), dtype=jnp.float32),
        packed_misc_i32=jnp.zeros((5,), dtype=jnp.int32),
        pages_tables=jnp.zeros((4, 2), dtype=jnp.int32),
        input_ids_buf=input_ids,
        position_ids_buf=jnp.arange(input_ids.shape[0], dtype=jnp.int32),
    )


def test_device_token_handoff_patches_flattened_input_ids_without_host_tokens():
    metadata = _metadata(jnp.array([11, 0, 22, 0, 33, 44], dtype=jnp.int32))
    handoff = DeviceInputTokenHandoff(
        input_positions=jnp.array([1, 3, 0, 0], dtype=jnp.int32),
        token_ids=jnp.array([101, 202, 0, 0], dtype=jnp.int32),
        count=jnp.array(2, dtype=jnp.int32),
    )

    patched = ExecutionManager._apply_device_token_handoff(metadata, handoff)

    np.testing.assert_array_equal(
        np.asarray(patched.model_input_ids),
        np.array([11, 101, 22, 202, 33, 44], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.input_ids_buf),
        np.array([11, 0, 22, 0, 33, 44], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(patched.input_ids_buf),
        np.array([11, 0, 22, 0, 33, 44], dtype=np.int32),
    )
    assert patched.position_ids_buf is metadata.position_ids_buf
