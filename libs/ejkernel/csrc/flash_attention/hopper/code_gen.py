"""Generate Hopper/SM90+ FlashAttention instantiation stubs.

This script mirrors upstream behavior while adding:
- deterministic ordering
- optional SM selection
- write-if-changed to avoid unnecessary rebuilds
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

DTYPE_MAP: Mapping[str, str] = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
    "e4m3": "cutlass::float_e4m3_t",
}

DTYPE_MAP_FWD_SM8X: Mapping[str, str] = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

DTYPE_MAP_BWD: Mapping[str, str] = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

DEFAULT_SMS = (80, 90, 100, 110, 120)
HEAD_DIMS = (64, 96, 128, 192, 256)
PAGED_KV = (False, True)
SPLIT = (False, True)
SOFTCAP = (False, True)
PACKGQA = (False, True)

PRELUDE = """
// Copyright (c) 2025, erfanzar.
//
// Original work by Tri Dao (Copyright (c) 2024).
// Special thanks to Tri Dao for the foundational implementation.
// We have only modified the code structure for better integration with minimal changes.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

"""
KERNEL_IMPL_FWD_SM90 = """#include \"flash_fwd_launch_template.h\"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<{ARCH}, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
#endif
"""  # noqa

KERNEL_IMPL_FWD_SM8X = """#include \"flash_fwd_launch_template.h\"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<80, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_<86, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
#endif
#endif
"""  # noqa

KERNEL_IMPL_BWD_SM90 = """#include \"flash_bwd_launch_template.h\"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template<>
void run_mha_bwd_<{ARCH}, {DTYPE}, {HEAD_DIM}, {SOFTCAP}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{ARCH}, {DTYPE}, {SOFTCAP}>(params, stream);
}}
#endif
"""

KERNEL_IMPL_BWD_SM8X = """#include \"flash_bwd_launch_template.h\"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template<>
void run_mha_bwd_<80, {DTYPE}, {HEAD_DIM}, {SOFTCAP}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<80, {DTYPE}, {SOFTCAP}>(params, stream);
}}
template<>
void run_mha_bwd_<86, {DTYPE}, {HEAD_DIM}, {SOFTCAP}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<86, {DTYPE}, {SOFTCAP}>(params, stream);
}}
#endif
#endif
"""


@dataclass(frozen=True)
class Kernel:
    """A single FlashAttention kernel instantiation variant for Hopper/SM8x.

    Attributes:
        sm: Target SM version (e.g. 80, 90).
        arch: Resolved architecture bucket (80 or 90).
        dtype: Data-type key (``"fp16"``, ``"bf16"``, or ``"e4m3"``).
        head_dim: Query/key head dimension.
        head_dim_v: Value head dimension.
        split: Whether this is a split-KV variant.
        paged_kv: Whether paged KV is enabled.
        softcap: Whether softcap logit clamping is enabled.
        packgqa: Whether packed GQA layout is enabled.
        direction: ``"fwd"`` or ``"bwd"``.
    """

    sm: int
    arch: int
    dtype: str
    head_dim: int
    head_dim_v: int
    split: bool
    paged_kv: bool
    softcap: bool
    packgqa: bool
    direction: str

    def render(self) -> str:
        """Render the C++ template instantiation source code for this kernel."""
        if self.direction == "fwd":
            if self.arch == 90:
                packgqa = self.packgqa or self.paged_kv or self.split
                return KERNEL_IMPL_FWD_SM90.format(
                    ARCH=str(self.arch),
                    DTYPE=DTYPE_MAP[self.dtype],
                    HEAD_DIM=self.head_dim,
                    HEAD_DIM_V=self.head_dim_v,
                    SPLIT=str(self.split).lower(),
                    PAGEDKV=str(self.paged_kv).lower(),
                    SOFTCAP=str(self.softcap).lower(),
                    PACKGQA=str(packgqa).lower(),
                )
            return KERNEL_IMPL_FWD_SM8X.format(
                DTYPE=DTYPE_MAP_FWD_SM8X[self.dtype],
                HEAD_DIM=self.head_dim,
                HEAD_DIM_V=self.head_dim_v,
                SPLIT=str(self.split).lower(),
                PAGEDKV=str(self.paged_kv).lower(),
                SOFTCAP=str(self.softcap).lower(),
                PACKGQA=str(True).lower(),
            )

        if self.direction == "bwd":
            if self.arch == 90:
                return KERNEL_IMPL_BWD_SM90.format(
                    ARCH=str(self.arch),
                    DTYPE=DTYPE_MAP_BWD[self.dtype],
                    HEAD_DIM=self.head_dim,
                    SOFTCAP=str(self.softcap).lower(),
                )
            return KERNEL_IMPL_BWD_SM8X.format(
                DTYPE=DTYPE_MAP_BWD[self.dtype],
                HEAD_DIM=self.head_dim,
                SOFTCAP=str(self.softcap).lower(),
            )

        raise ValueError(f"Unknown direction: {self.direction}")

    @property
    def filename(self) -> str:
        """Derive the output ``.cu`` filename from this kernel's parameters."""
        parts = ["flash", self.direction, f"hdim{self.head_dim}"]
        if self.head_dim_v != self.head_dim:
            parts.append(str(self.head_dim_v))
        parts.append(self.dtype)
        if self.paged_kv:
            parts.append("paged")
        if self.split:
            parts.append("split")
        if self.softcap:
            parts.append("softcap")
        if self.packgqa:
            parts.append("packgqa")
        return "_".join(parts) + f"_sm{self.sm}.cu"


def _write_if_changed(path: Path, content: str) -> None:
    """Write *content* to *path* only if it differs from the existing file."""
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def _resolve_arch(sm: int) -> int:
    """Map an SM version to its architecture bucket (80 or 90)."""
    return 90 if sm >= 90 else 80


def _iter_fwd_kernels(sms: Iterable[int]) -> Iterable[Kernel]:
    """Yield all forward-pass kernel variants for the given SM versions."""
    for sm, dtype, head_dim, paged_kv, split, softcap, packgqa in itertools.product(
        sms, DTYPE_MAP.keys(), HEAD_DIMS, PAGED_KV, SPLIT, SOFTCAP, PACKGQA
    ):
        arch = _resolve_arch(sm)
        if arch != 90 and dtype not in DTYPE_MAP_FWD_SM8X:
            continue
        head_dim_v = head_dim
        yield Kernel(
            sm=sm,
            arch=arch,
            dtype=dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            split=split,
            paged_kv=paged_kv,
            softcap=softcap,
            packgqa=packgqa,
            direction="fwd",
        )


def _iter_bwd_kernels(sms: Iterable[int]) -> Iterable[Kernel]:
    """Yield all backward-pass kernel variants for the given SM versions."""
    for sm, dtype, head_dim, softcap in itertools.product(sms, DTYPE_MAP_BWD.keys(), HEAD_DIMS, SOFTCAP):
        arch = _resolve_arch(sm)
        yield Kernel(
            sm=sm,
            arch=arch,
            dtype=dtype,
            head_dim=head_dim,
            head_dim_v=head_dim,
            split=False,
            paged_kv=False,
            softcap=softcap,
            packgqa=False,
            direction="bwd",
        )


def iter_kernels(sms: Iterable[int]) -> Iterable[Kernel]:
    """Yield all forward and backward kernel variants for the given SM versions."""
    yield from _iter_fwd_kernels(sms)
    yield from _iter_bwd_kernels(sms)


def generate(output_dir: Path, sms: Iterable[int]) -> int:
    """Write all generated ``.cu`` files into *output_dir*.

    Returns:
        The total number of files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for kernel in iter_kernels(sms):
        text = PRELUDE + kernel.render()
        _write_if_changed(output_dir / kernel.filename, text)
        count += 1
    return count


def main() -> None:
    """Parse CLI arguments and generate Hopper FlashAttention instantiation stubs."""
    parser = argparse.ArgumentParser(
        prog="code_gen",
        description="Generate Hopper/SM90+ FlashAttention instantiation stubs.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Output directory for generated .cu files (defaults to hopper/instantiations).",
    )
    parser.add_argument(
        "--sms",
        default=",".join(str(sm) for sm in DEFAULT_SMS),
        help="Comma-separated list of SM versions to emit (e.g., 80,90,100,110,120).",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "instantiations"
    sms = tuple(int(item.strip()) for item in args.sms.split(",") if item.strip())
    if not sms:
        sms = DEFAULT_SMS
    generated = generate(out_dir, sms)
    print(f"Generated {generated} files for SMs={sms} in {out_dir}")


if __name__ == "__main__":
    main()
