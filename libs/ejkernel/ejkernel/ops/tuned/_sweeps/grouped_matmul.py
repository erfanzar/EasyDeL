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

"""Sweep for the stacked-expert grouped matmul (MoE).

Measured on v5p at DeepSeek-V4 shapes: cost tracks the number of NON-EMPTY
experts rather than the token count, because each expert's group is far smaller
than one m-tile and therefore pays for a whole one. A shard's time was flat
within 0.1% across a 0.85x-2.0x token load at fixed buffer height, while going
from 4 to 48 touched experts moved it 0.039 -> 0.170 ms.

So ``block_m`` is what matters, and the quantity that sets it is rows per
expert (``m / num_groups``), not the buffer height. It is also not monotone —
at 48 rows/expert a tile of 8 was 45% *slower* than 64, because the MXU goes
idle — which is why this sweeps the tile rather than computing it.

Both the quantized (Pallas v3, int4/int8 codes + per-channel scales) and dense
paths are candidates at every point, so the table records which backend won
instead of assuming TPU means Pallas.
"""

from __future__ import annotations

import typing as tp

from .._store import bucket
from .._sweep import Candidate, SweepPoint, SweepSpec, register_sweep

KERNEL = "grouped_matmul"

#: Token counts spanning decode (a handful of rows per expert) through prefill.
DEFAULT_TOKENS = (8, 32, 128, 512, 2048, 8192)

#: Tile heights to consider. 8 is the Mosaic floor; past 128 the wasted rows
#: dominate at every MoE shape measured.
DEFAULT_BLOCK_M = (8, 16, 32, 64, 128, 256)


def _make_candidates(rows, kernel, scales, group_sizes, k_dim, n_dim, num_groups, block_ms):
    import jax
    from jax import numpy as jnp

    from ejkernel.modules import GroupedMatmulConfig, grouped_matmul

    aligned = (k_dim % 128 == 0) and (n_dim % 128 == 0)
    out: list[Candidate] = []

    for block_m in block_ms:
        cfg = (
            GroupedMatmulConfig(block_m=block_m, block_k=k_dim, block_n=n_dim)
            if aligned
            # block_k/n=0 defers to the kernel's own tiler, which handles
            # contractions Mosaic cannot slice on a 128 boundary.
            else GroupedMatmulConfig(block_m=block_m, block_k=0, block_n=0)
        )

        def build(_cfg=cfg):
            fn = jax.jit(
                lambda r, g: grouped_matmul(
                    r,
                    kernel,
                    g,
                    preferred_element_type=jnp.bfloat16,
                    **(
                        {"rhs_scale": scales.reshape(num_groups, 1, 1, n_dim), "use_v3": True}
                        if scales is not None
                        else {}
                    ),
                    platform="pallas",
                    cfg=_cfg,
                )
            )
            return lambda: fn(rows, group_sizes)

        out.append(Candidate("pallas", {"block_m": block_m}, build))

    def build_xla():
        import jax as _jax
        from jax import numpy as _jnp

        fn = _jax.jit(
            lambda r, g: grouped_matmul(r, kernel, g, preferred_element_type=_jnp.bfloat16, platform="xla")
        )
        return lambda: fn(rows, group_sizes)

    # Only meaningful for dense weights: the XLA path has no quantized variant.
    if scales is None:
        out.append(Candidate("xla", {}, build_xla))
    return out


def points(
    *,
    num_experts: int = 256,
    ep: int = 4,
    hidden: int = 4096,
    intermediate: int = 2048,
    top_k: int = 6,
    bits: int = 4,
    dense: bool = False,
    tokens: tp.Sequence[int] = DEFAULT_TOKENS,
    block_ms: tp.Sequence[int] = DEFAULT_BLOCK_M,
    **_: tp.Any,
) -> tp.Iterator[SweepPoint]:
    """Yield one point per token count for a given MoE geometry."""
    import numpy as np
    from jax import numpy as jnp

    e_local = max(1, num_experts // max(1, ep))
    rng = np.random.default_rng(0)
    weights = jnp.asarray(rng.standard_normal((e_local, hidden, intermediate)).astype(np.float32) * 0.02, jnp.bfloat16)

    if dense:
        kernel, scales, w_dtype = weights, None, "bfloat16"
    else:
        # Per-output-channel codes + scales, reducing over the contraction axis
        # -- the layout the v3 grouped matmul expects for `rhs_scale`. This is a
        # TIMING sweep, so only the shapes and dtypes have to be faithful; the
        # values never reach a correctness check. Done inline rather than
        # reaching for a framework quantizer, which would invert the dependency.
        qmax = 2 ** (bits - 1) - 1
        scales = jnp.max(jnp.abs(weights), axis=-2, keepdims=True) / qmax
        code_dtype = jnp.int4 if bits == 4 else jnp.int8
        kernel = jnp.clip(jnp.round(weights / scales), -qmax - 1, qmax).astype(code_dtype)
        w_dtype = f"int{bits}"
    del weights

    for tok in tokens:
        buf = tok * top_k
        rows = jnp.asarray(rng.standard_normal((buf, hidden)).astype(np.float32) * 0.1, jnp.bfloat16)
        # group_sizes must never sum past the buffer height. Claiming more rows
        # than exist does not raise -- it FAULTS THE DEVICE ("the program
        # continuator has halted unexpectedly"), after which every later point
        # in the sweep fails too, silently. At small token counts a shard owns
        # fewer rows than it has experts, so some groups are legitimately empty.
        owned = min(buf, max(1, buf // max(1, ep)))
        base, rem = divmod(owned, e_local)
        sizes = np.full(e_local, base, dtype=np.int64)
        sizes[:rem] += 1
        assert int(sizes.sum()) == owned <= buf, (int(sizes.sum()), owned, buf)
        group_sizes = jnp.asarray(sizes, jnp.int32)
        cands = _make_candidates(rows, kernel, scales, group_sizes, hidden, intermediate, e_local, block_ms)
        # The untuned default: `_expert_gmm`'s historical buffer-height rule.
        default_bm = 64 if buf <= 1024 else 128
        baseline = next((c for c in cands if c.config.get("block_m") == default_bm), None)
        yield SweepPoint(
            dtypes={"lhs": "bfloat16", "rhs": w_dtype},
            shape={"m": buf, "k": hidden, "n": intermediate, "e": num_experts},
            candidates=cands,
            baseline=baseline,
            label=(
                f"tokens={tok} m={buf} rows/expert={buf / num_experts:.2f} "
                f"bucket={bucket(buf)} default_block_m={default_bm}"
            ),
        )


register_sweep(
    SweepSpec(
        kernel=KERNEL,
        points=points,
        description="Stacked-expert grouped matmul: block_m vs rows-per-expert, Pallas vs XLA.",
    )
)
