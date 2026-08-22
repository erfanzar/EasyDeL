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

"""Sweeps driven by the repository's benchmark registry.

``benchmarks/_op_benchmark_registry.py`` already knows, for every op, how to
build valid inputs and which shapes are worth looking at. Rewriting that per
kernel would be both duplicated effort and a second source of truth, so a sweep
here is only two things: which config fields are tunable, and what values to try.
Everything else — inputs, shapes, the op callable — comes from the registry, so
any op it covers can be tuned by adding a few lines to :data:`TUNABLES`.

Developer tool: the benchmark registry lives beside the package rather than in
it, so this needs a source checkout. It fails with a clear message otherwise
rather than shipping a broken sweep.
"""

from __future__ import annotations

import itertools
import sys
import typing as tp
from pathlib import Path

from .._sweep import Candidate, SweepPoint, SweepSpec, register_sweep

#: Attention block sizes to try. Ranges past the sequence length simply fail to
#: compile and are dropped as "not an option here", so covering 1024/2048 costs
#: nothing at short sequences and finds the win at long ones.
_ATTN_BLOCKS = (64, 128, 256, 512, 1024, 2048)

#: Config fields worth sweeping per op, and the values to try.
#:
#: ``ragged_page_attention_*`` tune the KV-page and query block counts; the
#: grouped matmul tunes its m-tile, which is what rows-per-expert actually
#: sets; blocksparse/splash tunes its q/kv block sizes.
TUNABLES: dict[str, dict[str, list]] = {
    "ragged_page_attention_v3": {
        "num_kv_pages_per_block": [2, 4, 8, 16, 32],
        "num_queries_per_block": [8, 16, 32, 64, 128],
    },
    "ragged_page_attention_v2": {
        "num_kv_pages_per_block": [2, 4, 8, 16, 32],
        "num_queries_per_block": [8, 16, 32, 64, 128],
    },
    "grouped_matmul": {"block_m": [8, 16, 32, 64, 128, 256]},
    "grouped_matmul_v3": {"block_m": [8, 16, 32, 64, 128, 256]},
    # FwdParams/BwdParams name these `q_blocksize`/`kv_blocksize`; the TPU
    # heuristic defaults to 128/128 for both passes.
    "blocksparse_attention": {
        "fwd_params": [
            {"q_blocksize": q, "kv_blocksize": kv}
            for q, kv in itertools.product(_ATTN_BLOCKS, _ATTN_BLOCKS)
        ],
    },
    "blocksparse_attention_bwd": {
        "bwd_params": [
            {"q_blocksize": q, "kv_blocksize": kv}
            for q, kv in itertools.product(_ATTN_BLOCKS, _ATTN_BLOCKS)
        ],
    },
}

#: Ops whose sweep should time the BACKWARD pass. Tuning `bwd_params` against a
#: forward-only measurement would record whichever value happened to be fastest
#: at doing nothing.
BACKWARD_OPS: dict[str, str] = {"blocksparse_attention_bwd": "blocksparse_attention"}

#: Per-op default config, i.e. what the kernel picks with no tuned table.
DEFAULT_KNOBS: dict[str, dict] = {
    "blocksparse_attention": {"fwd_params": {"q_blocksize": 128, "kv_blocksize": 128}},
    "blocksparse_attention_bwd": {"bwd_params": {"q_blocksize": 128, "kv_blocksize": 128}},
}

#: Config class per op, resolved from ejkernel.modules.operations.configs.
CONFIG_CLASS: dict[str, str] = {
    "ragged_page_attention_v3": "RaggedPageAttentionv3Config",
    "ragged_page_attention_v2": "RaggedPageAttentionv2Config",
    "grouped_matmul": "GroupedMatmulConfig",
    "grouped_matmul_v3": "GroupedMatmulConfig",
    "blocksparse_attention": "BlockSparseAttentionConfig",
    "blocksparse_attention_bwd": "BlockSparseAttentionConfig",
}

#: Platforms to try. XLA is included so the table records which backend won
#: rather than assuming TPU implies Pallas.
PLATFORMS: dict[str, list[str]] = {
    "ragged_page_attention_v3": ["pallas"],
    "ragged_page_attention_v2": ["pallas"],
    "grouped_matmul": ["pallas", "xla"],
    "grouped_matmul_v3": ["pallas"],
    "blocksparse_attention": ["pallas"],
    "blocksparse_attention_bwd": ["pallas"],
}


def _load_registry():
    """Import the benchmark registry from the source checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "benchmarks" / "_op_benchmark_registry.py"
        if candidate.exists():
            if str(candidate.parent) not in sys.path:
                sys.path.insert(0, str(candidate.parent))
            import _op_benchmark_registry as reg  # type: ignore[import-not-found]

            return reg
    raise RuntimeError(
        "benchmark registry not found; sweeps need a source checkout of ejkernel "
        "(libs/ejkernel/benchmarks/_op_benchmark_registry.py)"
    )


def _default_knobs(op: str, config: dict) -> dict | None:
    """The config the kernel uses with NO tuned table.

    ragged_page_attention falls back to ``(2048 // page_size, 32)`` for TPU v5/v6
    (4096 // page_size on v7) -- see ``get_tuned_block_sizes``. Measuring it is
    what turns a row from "this candidate beat that one" into "tuning beat not
    tuning".
    """
    if op in DEFAULT_KNOBS:
        return dict(DEFAULT_KNOBS[op])
    if op.startswith("ragged_page_attention"):
        page_size = int(config.get("page_size") or 0)
        if page_size <= 0:
            return None
        import jax

        kind = str(getattr(jax.devices()[0], "device_kind", ""))
        per = 4096 if "v7" in kind else 2048
        return {"num_kv_pages_per_block": max(1, per // page_size), "num_queries_per_block": 32}
    return None


def _config_class(op: str):
    from ejkernel.modules.operations import configs as cfgmod

    return getattr(cfgmod, CONFIG_CLASS[op])


def _numeric_shape(config: dict[str, tp.Any]) -> dict[str, int]:
    """Shape key from a benchmark config: every positive integer field.

    Generic on purpose — each op's config names its own dimensions, and
    hand-mapping them per kernel is exactly the per-kernel schema this table
    exists to avoid. Values are bucketed to powers of two downstream.
    """
    out: dict[str, int] = {}
    for key, value in config.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            out[key] = value
    return out


#: Config fields that change the kernel's behaviour -- and therefore its best
#: block sizes -- without being a dtype or a dimension.
_VARIANT_FIELDS = ("causal", "sliding", "sliding_window", "optimized", "transpose_rhs")


def _dtypes_of(config: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Discriminator key: dtypes plus any behaviour-changing flags.

    These have to be part of the key. The blocksparse configs differ only in
    `causal`/`sliding`, so leaving them out collapsed 12 measured points into 4
    rows -- and would have handed a causal-tuned block size to a non-causal
    call.
    """
    out = {}
    for key, value in config.items():
        if "dtype" in key and value is not None:
            out[key] = value
    for key in _VARIANT_FIELDS:
        if key in config:
            out[key] = config[key]
    return out or {"compute": "bfloat16"}


#: Ops whose generator emits ``(q, k, v, causal, sliding_window)`` -- the last
#: two are KEYWORD arguments, which the registry routes via
#: ``_wrap_attention_like``. Passing them positionally lands them in the
#: ``q_segment_ids``/``kv_segment_ids`` slots and every candidate dies on a
#: jaxtyping error, which the sweep then reports as "no measurable points".
_ATTENTION_LIKE = {"blocksparse_attention", "blocksparse_attention_bwd"}


def _split_inputs(op: str, inputs: tuple) -> tuple[tuple, dict]:
    """Split a generator's tuple into differentiable primals and fixed kwargs."""
    if op in _ATTENTION_LIKE and len(inputs) >= 5:
        q, k, v, causal, sliding_window = inputs[:5]
        return (q, k, v), {"causal": causal, "sliding_window": sliding_window}
    return tuple(inputs), {}


def _builder(jax, spec, cfg_cls, platform, knob, inputs, static, backward, op=""):
    """Build a jitted zero-arg callable for one candidate.

    With *backward*, the timed function is the gradient w.r.t. the leading
    array inputs (q/k/v), which is what actually exercises ``bwd_params``.
    """
    primals, extra_kwargs = _split_inputs(op, inputs)
    call_kwargs = {**static, **extra_kwargs}
    n_grad = min(3, len(primals))

    def build():
        kernel_cfg = cfg_cls(**knob)

        def forward(*args):
            rest = primals[len(args) :]
            out = spec.op_fn(*args, *rest, platform=platform, cfg=kernel_cfg, **call_kwargs)
            return out[0] if isinstance(out, tuple) else out

        if not backward:
            return jax.jit(lambda: forward(*primals))

        def loss(*args):
            import jax.numpy as jnp

            return jnp.sum(forward(*args).astype(jnp.float32))

        grad_fn = jax.jit(jax.grad(loss, argnums=tuple(range(n_grad))))
        return lambda: grad_fn(*primals[:n_grad])

    return build


def _rpa_v3_key(config: dict) -> tuple[dict, dict] | None:
    """Build the key ragged_page_attention_v3 will actually look up with.

    A sweep keyed on the BENCHMARK config's field names (`dim`, `qheads`,
    `total_q`) writes rows the kernel can never find, because it queries with
    its own vocabulary (`head_dim`, `q_heads`, `max_len`) after normalizing
    heads for dtype packing and x2 KV interleaving. Reuse the kernel's own
    `get_lookup_keys` so the two agree by construction rather than by luck.
    """
    try:
        from jax import numpy as jnp

        from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._utils import get_lookup_keys
    except Exception:
        return None
    try:
        page_size = int(config["page_size"])
        _dev, page_size_k, dtypes_k, heads_k, max_len_k = get_lookup_keys(
            page_size,
            jnp.bfloat16,
            jnp.bfloat16,
            int(config["qheads"]),
            int(config["kvheads"]),
            int(config["dim"]),
            page_size * int(config["pages_per_seq"]),
        )
    except Exception:
        return None
    import re

    m = re.match(r"q_head-(\d+)_kv_head-(\d+)_head-(\d+)", heads_k)
    d = re.match(r"q_(.+?)_kv_(.+)$", dtypes_k)
    if not m or not d:
        return None
    return (
        {"q": d.group(1), "kv": d.group(2)},
        {
            "page_size": page_size_k,
            "q_heads": int(m.group(1)),
            "kv_heads": int(m.group(2)),
            "head_dim": int(m.group(3)),
            "max_len": max_len_k,
        },
    )


#: Ops that must key rows in the kernel's own lookup vocabulary.
KEY_BUILDERS: dict[str, tp.Callable[[dict], tuple[dict, dict] | None]] = {
    "ragged_page_attention_v3": _rpa_v3_key,
}


def make_points(
    op: str,
    *,
    limit: int | None = None,
    config_grid: dict[str, list] | None = None,
    knob_grid: dict[str, list] | None = None,
    **_: tp.Any,
) -> tp.Iterator[SweepPoint]:
    """Yield one point per config for *op*.

    Args:
        op: Registry key.
        limit: Keep only the first N configs.
        config_grid: Cartesian product of INPUT shapes to sweep instead of the
            registry's own configs. The registry's grids are sized for
            benchmarking, not tuning -- ragged_page_attention_v3 ships two --
            so a table usually wants its own coverage.
        knob_grid: Override the tunable values from :data:`TUNABLES`.
    """
    import jax

    reg = _load_registry()
    registry_op = BACKWARD_OPS.get(op, op)
    backward = op in BACKWARD_OPS
    spec = reg.SPECS.get(registry_op)
    if spec is None:
        raise SystemExit(f"{registry_op!r} is not in the benchmark registry; known: {sorted(reg.SPECS)}")
    cfg_cls = _config_class(op)
    knobs = dict(knob_grid or TUNABLES[op])
    platforms = PLATFORMS.get(op, ["pallas"])
    static_names = list(spec.static_kwargs or [])

    if config_grid:
        base = dict(spec.configs[0]) if spec.configs else {}
        keys = list(config_grid)
        configs = [{**base, **dict(zip(keys, vals, strict=True))} for vals in itertools.product(*config_grid.values())]
    else:
        configs = spec.configs
    configs = configs[:limit] if limit else configs
    for config in configs:
        try:
            inputs = spec.input_generator(config)
        except Exception as exc:
            print(f"  skip {config}: input generation failed ({type(exc).__name__}: {exc})", flush=True)
            continue
        static = {} if op in _ATTENTION_LIKE else {k: config[k] for k in static_names if k in config}

        candidates: list[Candidate] = []
        for platform in platforms:
            for values in itertools.product(*knobs.values()):
                knob = dict(zip(knobs, values, strict=True))

                candidates.append(
                    Candidate(
                        platform,
                        dict(knob),
                        _builder(jax, spec, cfg_cls, platform, knob, inputs, static, backward, op),
                    )
                )

        baseline = None
        default = _default_knobs(op, config)
        if default is not None:

            baseline = Candidate(
                platforms[0],
                dict(default),
                _builder(jax, spec, cfg_cls, platforms[0], default, inputs, static, backward, op),
            )

        built = KEY_BUILDERS.get(op, lambda _c: None)(config)
        if built is not None:
            key_dtypes, key_shape = built
        else:
            key_dtypes, key_shape = _dtypes_of(config), _numeric_shape(config)

        yield SweepPoint(
            dtypes=key_dtypes,
            shape=key_shape,
            candidates=candidates,
            baseline=baseline,
            label=f"{op} {config}",
        )


def _register(op: str, description: str) -> None:
    register_sweep(
        SweepSpec(
            kernel=op,
            points=lambda _op=op, **kw: make_points(_op, **kw),
            description=description,
        )
    )


_register("ragged_page_attention_v3", "Paged attention v3: KV-page/query block counts.")
_register("ragged_page_attention_v2", "Paged attention v2: KV-page/query block counts.")
_register("blocksparse_attention", "Splash/block-sparse attention (forward): q/kv block sizes.")
_register("blocksparse_attention_bwd", "Splash/block-sparse attention (backward): bwd q/kv block sizes.")
