"""Generate FlashAttention kernel instantiation stubs.

This script emits the per-head-dim CUDA instantiation files used by the
CUTLASS-based SM80 implementation. It is safe to re-run and will only
rewrite files when content changes.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

DTYPE_MAP: Mapping[str, str] = {"fp16": "cutlass::half_t", "bf16": "cutlass::bfloat16_t"}

DEFAULT_SMS = (80, 90, 100, 110, 120)
HEAD_DIMS = (32, 64, 96, 128, 192, 256)
IS_CAUSAL = ("false", "true")
NAMESPACE_INCLUDE = '#include "namespace_config.h"\n'

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


def _fwd_template() -> str:
    """Return the C++ forward-pass template string with format placeholders."""
    return (
        NAMESPACE_INCLUDE
        + """#include \"flash_fwd_launch_template.h\"

namespace FLASH_NAMESPACE {{

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}

}} // namespace FLASH_NAMESPACE"""
    )


def _fwd_split_template() -> str:
    """Return the C++ split-KV forward-pass template string."""
    return (
        NAMESPACE_INCLUDE
        + """#include \"flash_fwd_launch_template.h\"

namespace FLASH_NAMESPACE {{

template void run_mha_fwd_splitkv_dispatch<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream);

}} // namespace FLASH_NAMESPACE"""  # noqa
    )


def _bwd_template() -> str:
    """Return the C++ backward-pass template string."""
    return (
        NAMESPACE_INCLUDE
        + """#include \"flash_bwd_launch_template.h\"

namespace FLASH_NAMESPACE {{

template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}

}} // namespace FLASH_NAMESPACE"""
    )


@dataclass(frozen=True)
class Kernel:
    """A single SM80-family FlashAttention kernel instantiation variant.

    Attributes:
        sm: Target SM version.
        dtype: Data-type key (``"fp16"`` or ``"bf16"``).
        head_dim: Head dimension.
        is_causal: ``"true"`` or ``"false"`` string for the causal flag.
        direction: ``"fwd"``, ``"fwd_split"``, or ``"bwd"``.
    """

    sm: int
    dtype: str
    head_dim: int
    is_causal: str
    direction: str

    def render(self) -> str:
        """Render the C++ template instantiation source for this kernel."""
        templates = {
            "fwd": _fwd_template,
            "bwd": _bwd_template,
            "fwd_split": _fwd_split_template,
        }
        template = templates[self.direction]()
        return template.format(
            DTYPE=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim,
            IS_CAUSAL=self.is_causal,
        )

    @property
    def filename(self) -> str:
        """Derive the output ``.cu`` filename from this kernel's parameters."""
        causal = "causal_" if self.is_causal == "true" else ""
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_{causal}sm{self.sm}.cu"


def iter_kernels(sms: Iterable[int]) -> Iterable[Kernel]:
    """Yield all kernel variants (fwd, fwd_split, bwd) for the given SM versions."""
    for direction in ("fwd", "fwd_split", "bwd"):
        for dtype, head_dim, is_causal, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMS, IS_CAUSAL, sms):
            yield Kernel(
                sm=sm,
                dtype=dtype,
                head_dim=head_dim,
                is_causal=is_causal,
                direction=direction,
            )


def _write_if_changed(path: Path, content: str) -> None:
    """Write *content* to *path* only if it differs from the existing file."""
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


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
    """Parse CLI arguments and generate FlashAttention SM80 instantiation stubs."""
    parser = argparse.ArgumentParser(
        prog="code_gen",
        description="Generate FlashAttention kernel instantiation stubs.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Output directory for generated .cu files (defaults to this script's directory).",
    )
    parser.add_argument(
        "--sms",
        default="80,90,100,110,120",
        help="Comma-separated list of SM versions to emit (e.g., 80,90,100,110,120).",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    sms = tuple(int(item.strip()) for item in args.sms.split(",") if item.strip())
    if not sms:
        sms = DEFAULT_SMS
    generated = generate(out_dir, sms)
    print(f"Generated {generated} files for SMs={sms} in {out_dir}")


if __name__ == "__main__":
    main()
