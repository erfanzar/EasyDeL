import jax
import jax.numpy as jnp
import pytest
from ejkernel.kernels._pallas.tpu.grouped_matmulv3._pallas_impl import grouped_matmulv3_pallas_impl as gmm


@pytest.mark.parametrize(
    "bad",
    [
        "missing_row",
        "missing_channel",
        "row_shape",
        "channel_shape",
        "row_dtype",
        "channel_dtype",
        "lhs_dtype",
        "rhs_dtype",
        "acc_dtype",
        "out_dtype",
        "rhs_scale",
        "rhs_bias",
        "fuse_act",
    ],
)
def test_epilogue_rejects_invalid_combinations_before_hardware(bad):
    x = jax.ShapeDtypeStruct((64, 128), jnp.int8)
    w = jax.ShapeDtypeStruct((3, 128, 128), jnp.int4)
    g = jax.ShapeDtypeStruct((3,), jnp.int32)
    r = jax.ShapeDtypeStruct((64, 1), jnp.float32)
    s = jax.ShapeDtypeStruct((3, 1, 128), jnp.float32)
    options = dict(output_row_scale=r, output_channel_scale=s, preferred_element_type=jnp.bfloat16, acc_dtype=jnp.int32)
    if bad == "missing_row":
        options["output_row_scale"] = None
    elif bad == "missing_channel":
        options["output_channel_scale"] = None
    elif bad == "row_shape":
        options["output_row_scale"] = jax.ShapeDtypeStruct((64,), jnp.float32)
    elif bad == "channel_shape":
        options["output_channel_scale"] = jax.ShapeDtypeStruct((3, 128), jnp.float32)
    elif bad == "row_dtype":
        options["output_row_scale"] = jax.ShapeDtypeStruct((64, 1), jnp.bfloat16)
    elif bad == "channel_dtype":
        options["output_channel_scale"] = jax.ShapeDtypeStruct((3, 1, 128), jnp.bfloat16)
    elif bad == "lhs_dtype":
        x = jax.ShapeDtypeStruct((64, 128), jnp.bfloat16)
    elif bad == "rhs_dtype":
        w = jax.ShapeDtypeStruct((3, 128, 128), jnp.bfloat16)
    elif bad == "acc_dtype":
        options["acc_dtype"] = jnp.float32
    elif bad == "out_dtype":
        options["preferred_element_type"] = jnp.int32
    elif bad == "rhs_scale":
        options["rhs_scale"] = jax.ShapeDtypeStruct((3, 1, 1, 128), jnp.float32)
    elif bad == "rhs_bias":
        options["rhs_bias"] = s
    elif bad == "fuse_act":
        options["fuse_act"] = "silu"
    with pytest.raises(ValueError, match=r"output scal|output_row_scale|output_channel_scale"):
        gmm.lower(x, w, g, **options)
