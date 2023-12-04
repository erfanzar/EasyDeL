from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
import jax.lax as lax
from flax.linen.dtypes import promote_dtype
from jax import vmap
from flax.linen import Dense, compact
from .utils import array_from_8bit, array_to_bit8


def get_tile_inds(format_):
    """
    The get_tile_inds function is used to get the indices of a tile in a tensor.

    :param format_: Determine the shape of the tile_inds array
    :return: The indices of the tiles in a given format
    
    """
    transform = lambda x: lax.cond(
        jnp.bitwise_and(jnp.array(format_), 1),
        lambda _: lax.transpose(x),
        lambda _: x,
        operand=None,
    )
    inverse_transform = lambda x: lax.cond(
        jnp.bitwise_and(jnp.array(format_), 1),
        lambda _: lax.transpose(x),
        lambda _: x,
        operand=None,
    )
    tile_size = _get_tile_size(format_)
    tile_inds = lax.broadcasted_iota(jnp.int32, [tile_size] * len(tile_size))
    tile_inds = inverse_transform(tile_inds)
    return tile_inds


@vmap
def _get_tile_size(format_):
    if isinstance(format_, str):
        return tuple(int(c) for c in format_)
    return format_


@dataclass
class MatmulLtState:
    _tile_indices: Optional[jax.Array] = None
    force_no_igemmlt: bool = False
    CB = None
    CxB = None
    SB = None
    SCB = None

    CxBt = None
    SBt = None
    CBt = None

    subB = None

    outlier_pool = None
    has_accumulated_gradients = False
    threshold = 0.0
    idx = None
    is_training = True
    has_fp16_weights = True
    memory_efficient_backward = False
    use_pool = False
    formatB = "col_turing"

    def reset_grads(self):
        """
        The reset_grads function is used to reset the gradients of all the parameters in our model.
        This function is called after each iteration of training, and before we update our weights.
        The reason for this is that if we don't reset these values, they will accumulate over time and
        cause problems with our weight updates.

        :param self: Represent the instance of the class
        :return: Nothing
        
        """
        self.CB = None
        self.CxB = None
        self.SB = None
        self.SCB = None

        self.CxBt = None
        self.SBt = None
        self.CBt = None

    @property
    def tile_indices(self):
        if self._tile_indices is None:
            self._tile_indices = get_tile_inds(self.formatB)
        return self._tile_indices


class Dense8Bit(Dense):
    @compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        if inputs.dtype == jnp.int8:
            inputs = array_from_8bit(inputs, self.dtype)
        kernel = array_from_8bit(self.param(
            'kernel',
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        ), self.param_dtype)
        if self.use_bias:
            bias = array_from_8bit(self.param(
                'bias', self.bias_init, (self.features,), self.param_dtype
            ), self.param_dtype)
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = lax.dot_general
        y = dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return array_to_bit8(y)


def matmul_true_int8(lhs, rhs):
    assert lhs.dtype == jnp.int8
    assert rhs.dtype == jnp.int8
    result = jnp.matmul(lhs, rhs, preferred_element_type=jnp.int32)
    assert result.dtype == jnp.int32
    return result


def aqt_matmul_int8(a, w):
    """
    The aqt_matmul_int8 function performs the following steps:

    :param a: Scale the input tensor
    :param w: Store the weights of the model
    :return: A result that is close to the original matrix multiplication
    
    """
    max_int8 = 127

    # This function is customizable and injectable, i.e:
    # users can inject custom quant code into an AQT config.
    def quant_int8(x):
        return jnp.clip(jnp.round(x), -max_int8, max_int8).astype(jnp.int8)

    # Calibration. Calibration function is also customizable and injectable.
    a_s = max_int8 / jnp.max(jnp.abs(a), axis=1, keepdims=True)
    w_s = max_int8 / jnp.max(jnp.abs(w), axis=0, keepdims=True)

    # int8 matmul with int32 accumulator
    result = matmul_true_int8(quant_int8(a * a_s), quant_int8(w * w_s)) / (a_s * w_s)

    return result
