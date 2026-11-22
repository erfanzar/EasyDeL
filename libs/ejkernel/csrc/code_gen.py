"""Run all kernel code generators in csrc/.

This script is a convenience wrapper around per-kernel code_gen.py files.
It is safe to re-run and forwards --sms to generators that support it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodegenTarget:
    """Description of a single per-kernel code generator to invoke.

    Attributes:
        name: Human-readable label printed during execution.
        path: Absolute path to the ``code_gen.py`` script.
        supports_sms: Whether this generator accepts a ``--sms`` flag.
        sms_override: If set, use this SM string instead of the global one.
    """

    name: str
    path: Path
    supports_sms: bool = False
    sms_override: str | None = None


def _repo_root() -> Path:
    """Return the repository root directory (two levels above this file)."""
    return Path(__file__).resolve().parent.parent


def _default_targets() -> list[CodegenTarget]:
    """Return the list of all known kernel code-generation targets."""
    root = _repo_root()
    return [
        CodegenTarget(
            name="flash_attention (sm80+)",
            path=root / "csrc" / "flash_attention" / "src" / "code_gen.py",
            supports_sms=True,
        ),
        CodegenTarget(
            name="flash_attention_hopper",
            path=root / "csrc" / "flash_attention" / "hopper" / "code_gen.py",
            supports_sms=True,
        ),
        CodegenTarget(
            name="ragged_page_attention_v3",
            path=root / "csrc" / "ragged_page_attention_v3" / "src" / "code_gen.py",
        ),
        CodegenTarget(
            name="blocksparse_attention",
            path=root / "csrc" / "blocksparse_attention" / "src" / "code_gen.py",
        ),
        CodegenTarget(
            name="quantized_matmul",
            path=root / "csrc" / "quantized_matmul" / "src" / "code_gen.py",
        ),
        CodegenTarget(
            name="unified_attention",
            path=root / "csrc" / "unified_attention" / "src" / "code_gen.py",
            supports_sms=True,
        ),
    ]


def _run_target(target: CodegenTarget, sms: str | None) -> None:
    """Execute a single code-generation target as a subprocess.

    Args:
        target: The codegen target to run.
        sms: Comma-separated SM versions to pass via ``--sms``, or ``None``
            to omit the flag.
    """
    if not target.path.exists():
        print(f"[skip] {target.name}: missing {target.path}", file=sys.stderr)
        return
    cmd = [sys.executable, str(target.path)]
    if target.supports_sms and sms is not None:
        sms_value = target.sms_override if target.sms_override is not None else sms
        if sms_value:
            cmd.extend(["--sms", sms_value])
    print(f"[run ] {target.name}")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Parse CLI arguments and invoke every registered code generator."""
    parser = argparse.ArgumentParser(
        prog="code_gen",
        description="Run all csrc code generators.",
    )
    parser.add_argument(
        "--sms",
        default="80,90,100,110,120",
        help="Comma-separated list of SM versions to emit where supported.",
    )
    parser.add_argument(
        "--no-sms",
        action="store_true",
        help="Do not pass --sms to generators (use their defaults).",
    )
    args = parser.parse_args()

    sms = None if args.no_sms else args.sms
    targets = _default_targets()

    failed = []
    for target in targets:
        try:
            _run_target(target, sms)
        except subprocess.CalledProcessError as exc:
            failed.append((target.name, exc.returncode))

    if failed:
        msg = ", ".join(f"{name} (exit {code})" for name, code in failed)
        raise SystemExit(f"code_gen failed: {msg}")


if __name__ == "__main__":
    main()
