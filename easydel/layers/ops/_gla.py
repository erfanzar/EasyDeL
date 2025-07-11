# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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


import jax
import jax.numpy as jnp
from jax import lax

from easydel.utils.compiling_utils import ejit

from ._base_operation import BaseOperation


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


@ejit(static_argnames=("chunk_size", "output_final_state", "dtype", "scale"))
def recurrent_gla(
    query: jnp.ndarray,  # shape: (B, S, H, D)
    key: jnp.ndarray,  # shape: (B, S, H, D)
    value: jnp.ndarray,  # shape: (B, S, H, V)
    gk: jnp.ndarray,  # shape: (B, S, H, D)
    scale: float = -1.0,
    initial_state: jnp.ndarray | None = None,  # shape: (B, H, D, V)
    chunk_size: int = 0,  # if > 0, process sequence in chunks
    dtype: jnp.dtype = jnp.float32,
    output_final_state: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Recurrent Gated Linear Attention with optional chunked sequence processing.

    This function implements a recurrent variant of gated linear attention, which processes
    the input sequence either as a whole or in chunks. The recurrent nature allows the model
    to maintain and update a hidden state throughout the sequence processing.

    Args:
        q: Query tensor of shape (B, S, H, D) where:
            - B is the batch size
            - S is the sequence length
            - H is the number of heads
            - D is the head dimension for queries and keys
        k: Key tensor of shape (B, S, H, D)
        v: Value tensor of shape (B, S, H, V) where:
            - V is the head dimension for values (can be different from D)
        gk: Gating tensor of shape (B, S, H, D), typically log-sigmoid values
        scale: Scaling factor for attention scores. If -1.0, uses 1/sqrt(D).
            Default: -1.0
        initial_state: Optional initial hidden state of shape (B, H, D, V).
            If None, initializes to zeros. Default: None
        chunk_size: Size of chunks for processing long sequences. If 0, processes
            the entire sequence at once. Default: 0
        dtype: Data type for computation. Default: jnp.float32
        output_final_state: Whether to return the final hidden state.
            Default: False

    Returns:
        A tuple (output, final_state) where:
        - output: Tensor of shape (B, S, H, V) containing the attention output
        - final_state: If output_final_state is True, returns the final hidden
          state of shape (B, H, D, V). Otherwise, returns None.

    Example:
        >>> B, S, H, D, V = 1, 32, 4, 64, 32
        >>> q = jax.random.normal(key, (B, S, H, D))
        >>> k = jax.random.normal(key, (B, S, H, D))
        >>> v = jax.random.normal(key, (B, S, H, V))
        >>> gk = jax.nn.log_sigmoid(jax.random.normal(key, (B, S, H, D)))
        >>> output, _ = recurrent_gla(q, k, v, gk, chunk_size=8)

    Notes:
        - The function is JIT-compiled with static arguments for chunk_size,
          output_final_state, and dtype.
        - For long sequences, using chunk_size > 0 can help manage memory usage
          by processing the sequence in smaller chunks.
        - The gating mechanism (gk) helps control the flow of information through
          the recurrent updates.
        - The hidden state maintains the running computation across the sequence
          or chunk boundaries.

    Implementation Details:
        The attention computation for each position i follows:
        1. q_i = q[i] * scale
        2. gk_i = exp(gk[i])
        3. kv_i = k[i] ⊗ v[i]  # outer product
        4. h = h * gk_i + kv_i  # recurrent update
        5. o_i = sum(q_i * h)   # output computation

    Memory Complexity:
        - Without chunking: O(B * S * H * max(D, V))
        - With chunking: O(B * chunk_size * H * max(D, V))
    """
    B, S, H, D = query.shape
    V = value.shape[-1]

    query = query.astype(dtype)
    key = key.astype(dtype)
    value = value.astype(dtype)
    gk = gk.astype(dtype)

    # Set scale if not provided
    if scale == -1.0:
        scale = D**-0.5

    # Initialize hidden state
    h = jnp.zeros((B, H, D, V), dtype=dtype)
    if initial_state is not None:
        h = h + initial_state.astype(dtype)

    def process_chunk(
        h: jnp.ndarray,
        chunk_idx: int,
        chunk_size: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, S)

        # Get chunk tensors
        q_chunk = query[:, start_idx:end_idx]
        k_chunk = key[:, start_idx:end_idx]
        v_chunk = value[:, start_idx:end_idx]
        gk_chunk = gk[:, start_idx:end_idx]

        def scan_fn(carry, x):
            (h,) = carry
            q_i, k_i, v_i, gk_i = x

            q_i = q_i * scale  # (B, H, D)
            gk_i = jnp.exp(gk_i)  # (B, H, D)
            kv_i = k_i[..., None] * v_i[..., None, :]  # (B, H, D, V)
            h = h * gk_i[..., None] + kv_i  # (B, H, D, V)
            o_i = jnp.sum(q_i[..., None] * h, axis=-2)  # (B, H, V)

            return (h,), o_i

        scan_inputs = (q_chunk, k_chunk, v_chunk, gk_chunk)

        (h_next,), o_chunk = lax.scan(
            scan_fn,
            (h,),
            [x.transpose(1, 0, 2, 3) for x in scan_inputs],
        )
        o_chunk = o_chunk.transpose(1, 0, 2, 3)

        return h_next, o_chunk

    if chunk_size > 0:
        num_chunks = ceildiv(S, chunk_size)
        o_chunks = []

        for chunk_idx in range(num_chunks):
            h, o_chunk = process_chunk(h, chunk_idx, chunk_size)
            o_chunks.append(o_chunk)

        o = jnp.concatenate(o_chunks, axis=1)
    else:

        def scan_fn(carry, x):
            (h,) = carry
            q_i, k_i, v_i, gk_i = x

            q_i = q_i * scale
            gk_i = jnp.exp(gk_i)
            kv_i = k_i[..., None] * v_i[..., None, :]
            h = h * gk_i[..., None] + kv_i
            o_i = jnp.sum(q_i[..., None] * h, axis=-2)

            return (h,), o_i

        scan_inputs = (query, key, value, gk)
        (h,), o = lax.scan(
            scan_fn,
            (h,),
            [x.transpose(1, 0, 2, 3) for x in scan_inputs],
        )
        o = o.transpose(1, 0, 2, 3)

    if not output_final_state:
        h = None

    return o, h


class RecurrentGLA(BaseOperation):
    def forward_native(
        self,
        query: jnp.ndarray,  # shape: (B, S, H, D)
        key: jnp.ndarray,  # shape: (B, S, H, D)
        value: jnp.ndarray,  # shape: (B, S, H, V)
        gk: jnp.ndarray,  # shape: (B, S, H, D)
        scale: float = -1.0,
        initial_state: jnp.ndarray | None = None,  # shape: (B, H, D, V)
        chunk_size: int = 0,  # if > 0, process sequence in chunks
        dtype: jnp.dtype = jnp.float32,
        output_final_state: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        return recurrent_gla(
            query=query,
            key=key,
            value=value,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            chunk_size=chunk_size,
            dtype=dtype,
            output_final_state=output_final_state,
        )


def test_recurrent_gla():
    # Test dimensions
    B, S, H, D, V = 1, 32, 4, 64, 32

    # Generate random inputs
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 5)

    query = jax.random.normal(keys[0], (B, S, H, D))
    key = jax.random.normal(keys[1], (B, S, H, D))
    value = jax.random.normal(keys[2], (B, S, H, V))
    gk = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, S, H, D)))
    h0 = jax.random.normal(keys[4], (B, H, D, V))

    # Test case 1: Without chunking
    o1, h1 = RecurrentGLA()(
        query,
        key,
        value,
        gk,
        initial_state=h0,
        output_final_state=True,
    )

    # Test case 2: With chunking
    chunk_size = 8
    o2, h2 = recurrent_gla(
        query,
        key,
        value,
        gk,
        initial_state=h0,
        chunk_size=chunk_size,
        output_final_state=True,
    )

    # Check shapes
    assert o1.shape == (B, S, H, V)
    assert o2.shape == (B, S, H, V)
    assert h1.shape == (B, H, D, V)
    assert h2.shape == (B, H, D, V)

    # Check that results are close
    assert jnp.allclose(o1, o2, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(h1, h2, rtol=1e-5, atol=1e-5)

    print("All tests passed!")


if __name__ == "__main__":
    test_recurrent_gla()
