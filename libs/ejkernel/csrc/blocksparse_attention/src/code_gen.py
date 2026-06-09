"""Generate blocksparse attention CUDA kernel instantiation stubs.

Emits per-head-dim, per-dtype, per-SM ``.cu`` files that explicitly
instantiate the blocksparse forward-pass template.  It is safe to re-run.
"""

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

DTYPE_MAP = {
    "fp16": "half",
    "bf16": "__nv_bfloat16",
}

SM = [80, 90, 100, 110, 120]
HEAD_DIMENSIONS = [32, 64, 96, 128, 192, 256]


def get_fwd_template() -> str:
    """Return the C++ forward-pass template string with format placeholders."""
    return """// Copyright (c) 2025, erfanzar.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "blocksparse_attention_launch_template.h"

namespace blocksparse {{

template<>
void run_blocksparse_fwd_<{DTYPE}, {QK_HEAD_DIM}, {V_HEAD_DIM}>(Blocksparse_fwd_params &params, cudaStream_t stream) {{
    run_blocksparse_fwd_hdim<{DTYPE}, {QK_HEAD_DIM}, {V_HEAD_DIM}>(params, stream);
}}

}} // namespace blocksparse
"""  # noga


@dataclass
class Kernel:
    """A single blocksparse attention kernel instantiation variant.

    Attributes:
        sm: Target SM version.
        dtype: Data-type key (``"fp16"`` or ``"bf16"``).
        qk_head_dim: Query/key head dimension.
        v_head_dim: Value head dimension.
    """

    sm: int
    dtype: str
    qk_head_dim: int
    v_head_dim: int

    @property
    def template(self) -> str:
        """Render the C++ source for this kernel variant."""
        return get_fwd_template().format(
            DTYPE=DTYPE_MAP[self.dtype],
            QK_HEAD_DIM=self.qk_head_dim,
            V_HEAD_DIM=self.v_head_dim,
        )

    @property
    def filename(self) -> str:
        """Derive the output ``.cu`` filename from this kernel's parameters."""
        return f"blocksparse_fwd_hdim{self.qk_head_dim}_vhdim{self.v_head_dim}_{self.dtype}_sm{self.sm}.cu"


def get_all_kernels() -> list[Kernel]:  # type: ignore
    """Yield all kernel variants across dtypes, head dimensions, and SM versions."""
    for dtype, head_dim, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SM):
        yield Kernel(
            sm=sm,
            dtype=dtype,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
        )


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    """Write a single kernel's ``.cu`` file into *autogen_dir*."""
    content = kernel.template
    (autogen_dir / kernel.filename).write_text(content)


def main(output_dir: str | None) -> None:
    """Generate all blocksparse kernel instantiation files.

    Args:
        output_dir: Target directory; defaults to this script's directory.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="code_gen",
        description="Generate the blocksparse_attention kernel instantiations",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels (defaults to current directory)",
    )
    args = parser.parse_args()
    main(args.output_dir)
