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

import jax
import jax.numpy as jnp
import pytest

from ejkernel.callib._cute_call import CAN_USE_CUTE, cute_call
from ejkernel.callib._cute_ffi import build_cute_ffi_call, has_cute_ffi_support

if CAN_USE_CUTE:
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

pytestmark = pytest.mark.skipif(not CAN_USE_CUTE, reason="CUTLASS CuTe DSL not installed")


if CAN_USE_CUTE:

    @cute.kernel
    def _copy_kernel(inp: cute.Tensor, out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < inp.shape[0]:
            out[tidx] = inp[tidx]

    @cute.jit
    def _copy_launch(stream: cuda.CUstream, inp: cute.Tensor, out: cute.Tensor):
        _copy_kernel(inp, out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def test_cute_call_rejects_legacy_jax_call_keyword():
    with pytest.raises(TypeError, match="jax_call"):
        _ = cute_call(
            jnp.ones((8,), dtype=jnp.float32),
            call=lambda x: x,
            out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
            jax_call=lambda x: x,
        )


@pytest.mark.skipif(not has_cute_ffi_support(), reason="CuTe primitive support unavailable")
def test_cute_call_jit_uses_primitive_call():
    primitive_call = build_cute_ffi_call(
        _copy_launch,
        output_shape_dtype=jax.ShapeDtypeStruct((8,), jnp.float16),
        compile_options="--enable-tvm-ffi",
    )

    @jax.jit
    def run(x):
        return cute_call(
            x,
            call=primitive_call,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )

    x = jnp.arange(8, dtype=jnp.float16)
    out = run(x)
    assert jnp.allclose(out, x)
