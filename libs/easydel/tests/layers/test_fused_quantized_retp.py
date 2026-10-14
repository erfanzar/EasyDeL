# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``retp_fused_state`` must re-interleave *quantized* fused leaves.

A dense fused projection keeps its data in ``.weight``; a quantized one has no
``.weight`` at all and stores the same fused axis in ``quant_kernel`` /
``quant_scales`` (plus optional ``quant_biases`` and the packed int4
companion). ``_fused_leaf_markers`` originally listed only ``.weight`` and
``.bias``, so every quantized fused projection was silently skipped while
``fused_param_tp`` was still stamped to the live tp -- the runtime splitter was
told the weight was TP-interleaved when the stored codes were still contiguous.

That is the identity at tp=1 and scrambles gate/up at tp>1. It was found on
DeepSeek-V4-Flash at A16W4 on a v5p-8: every shared expert diverged by rel ~1.3
at tp=2 (attention, which has no fused projections, stayed clean), and dumping
the stored ``quant_kernel`` showed it bit-identical at tp=1 and tp=2 -- i.e.
never re-interleaved.

These cases pin the transform at the leaf level, which is where the bug lived;
they need no mesh and no TPU.
"""

import numpy as np
import pytest
from easydel.layers.layouts._canonical import _fused_leaf_markers, _transform_last_axis

MODULE = "model.layers.0.mlp.shared_experts.gate_up_proj"


@pytest.mark.parametrize(
    "suffix",
    [".weight", ".bias", ".quant_kernel", ".quant_scales", ".quant_biases", ".quant_kernel_packed"],
)
def test_quantized_fused_leaves_are_matched(suffix):
    """Every leaf that can hold a fused projection's data must be matched.

    Missing one does not raise -- it leaves that leaf contiguous while the
    stamped ``fused_param_tp`` claims it is interleaved.
    """
    markers = _fused_leaf_markers(MODULE)
    assert f"{MODULE}{suffix}" in markers, f"{suffix} not matched by the fused-leaf markers"
    # the loader also sees `parameters.`-prefixed and `.value`-suffixed forms
    assert f"parameters.{MODULE}{suffix}" in markers
    assert f"{MODULE}{suffix}.value" in markers


@pytest.mark.parametrize("tp_size", [2, 4])
def test_interleave_round_trips_and_is_not_identity(tp_size):
    """canonical -> interleaved -> canonical, and the interleave must do work.

    The round-trip alone would pass trivially if the transform were a no-op,
    which is exactly the failure mode being guarded, so assert it actually
    permutes.
    """
    half = 8 * tp_size
    sizes = (half, half)
    x = np.arange(2 * 3 * (2 * half), dtype=np.float32).reshape(2, 3, 2 * half)

    interleaved = _transform_last_axis(x, sizes, tp_size, to_canonical=False)
    assert not np.array_equal(interleaved, x), "interleave must permute the fused axis"

    back = _transform_last_axis(interleaved, sizes, tp_size, to_canonical=True)
    assert np.array_equal(back, x), "de-interleave must invert the interleave"


@pytest.mark.parametrize("tp_size", [2, 4])
def test_interleave_groups_each_ranks_segments_together(tp_size):
    """The point of the layout: rank r must own gate_r and up_r contiguously.

    This is what lets a column-parallel shard hold a whole (gate, up) pair, and
    it is the property the runtime splitter relies on.
    """
    half = 4 * tp_size
    gate = np.arange(half, dtype=np.float32)
    up = 1000 + np.arange(half, dtype=np.float32)
    x = np.concatenate([gate, up])[None, :]

    out = _transform_last_axis(x, (half, half), tp_size, to_canonical=False)[0]

    local = half // tp_size
    for rank in range(tp_size):
        block = out[rank * 2 * local : (rank + 1) * 2 * local]
        assert np.array_equal(block[:local], gate[rank * local : (rank + 1) * local])
        assert np.array_equal(block[local:], up[rank * local : (rank + 1) * local])


def test_scales_shaped_leaf_is_transformed_like_the_kernel():
    """Channelwise scales are ``[1, N]``: same fused axis, same permutation.

    Re-interleaving codes without their per-output-channel scales would pair
    each column with another column's scale.
    """
    tp_size, half = 2, 8
    codes = np.arange(4 * 2 * half, dtype=np.float32).reshape(4, 2 * half)
    scales = np.arange(2 * half, dtype=np.float32).reshape(1, 2 * half)

    ck = _transform_last_axis(codes, (half, half), tp_size, to_canonical=False)
    sc = _transform_last_axis(scales, (half, half), tp_size, to_canonical=False)

    assert sc.shape == scales.shape

    # Recover the permutation the transform applies by running it on an index
    # vector, then require BOTH leaves to follow exactly that permutation --
    # otherwise a column's codes would be paired with another column's scale.
    perm = _transform_last_axis(
        np.arange(2 * half, dtype=np.float32)[None, :], (half, half), tp_size, to_canonical=False
    )[0].astype(int)
    assert np.array_equal(sc[0], scales[0][perm])
    for row in range(codes.shape[0]):
        assert np.array_equal(ck[row], codes[row][perm])


def test_non_fused_shaped_leaf_is_left_alone():
    """A leaf whose last axis is not the fused axis must pass through."""
    x = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    out = _transform_last_axis(x, (8, 8), 2, to_canonical=False)
    assert out is x


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
