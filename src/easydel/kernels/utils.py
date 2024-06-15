import jax.numpy
from jax.experimental import pallas as pl

next_power_of_2 = pl.next_power_of_2
MAX_FUSED_SIZE = 65536


def calculate_settings(cols, dtype: jax.numpy.dtype):
    max_fused_size = MAX_FUSED_SIZE // dtype.itemsize  # 4 o 2
    block_size = min(max_fused_size, next_power_of_2(cols))
    block_size = min(max(block_size, 128), 4096)
    num_warps = min(max(block_size // 256, 1), 8)
    return block_size, num_warps


def get_stride(array):
    strides = [1]
    for i in range(len(array.shape) - 1, 0, -1):  # Iterate in reverse shape order
        strides.insert(0, strides[0] * array.shape[i])
    return strides
