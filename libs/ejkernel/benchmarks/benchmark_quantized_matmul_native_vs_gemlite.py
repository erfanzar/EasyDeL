#!/usr/bin/env python3
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

"""Benchmark native ejkernel QMM backends against GemLite.

Default target:
    MxK @ KxN = 4096x8192 @ 8192x4096, 2-bit affine, group_size=128,
    bf16 activations, axis='col'.

Native ejkernel backends are called through the public operation layer with
strict_fuse=True and allow_dense_fallback=False. GemLite is only reported for
the comparable affine column-packed integer modes it supports (1/2/4/8 bit).
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import torch

from ejkernel.modules.operations.configs import QuantizedMatmulConfig
from ejkernel.modules.operations.quantized_matmul import quantized_matmul
from ejkernel.quantization._utils.bitpack import _pack_bits
from ejkernel.quantization._utils.qparams import resolve_qparams


@dataclass(frozen=True)
class Timing:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


CSV_FIELDS = (
    "workload",
    "backend",
    "mode",
    "axis",
    "bits",
    "group_size",
    "inner_iters",
    "mean_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "speedup_vs_gemlite",
)


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _int_csv(value: str) -> list[int]:
    if value == "all":
        return list(range(1, 9))
    return [int(part) for part in _csv(value)]


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _gemlite_dtype(name: str):
    from gemlite import DType

    if name == "bf16":
        return DType.BF16
    if name == "fp16":
        return DType.FP16
    if name == "fp32":
        return DType.FP32
    raise ValueError(f"unsupported GemLite dtype {name!r}")


def _time_jax(fn: Callable[[], jax.Array], warmup: int, iters: int, inner_iters: int = 1) -> Timing:
    y = fn()
    y.block_until_ready()
    for _ in range(warmup):
        for _ in range(inner_iters):
            y = fn()
        y.block_until_ready()
    vals: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        for _ in range(inner_iters):
            y = fn()
        y.block_until_ready()
        vals.append((time.perf_counter() - start) * 1000.0 / inner_iters)
    return Timing(
        mean_ms=statistics.fmean(vals),
        median_ms=statistics.median(vals),
        min_ms=min(vals),
        max_ms=max(vals),
    )


def _time_torch(fn: Callable[[], torch.Tensor], warmup: int, iters: int, inner_iters: int = 1) -> Timing:
    for _ in range(warmup):
        for _ in range(inner_iters):
            fn()
        torch.cuda.synchronize()
    vals: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(start.elapsed_time(end) / inner_iters)
    return Timing(
        mean_ms=statistics.fmean(vals),
        median_ms=statistics.median(vals),
        min_ms=min(vals),
        max_ms=max(vals),
    )


def _gemlite_auto_matmul_type(m: int, group_size: int) -> str:
    if m == 1:
        return "GEMV_SPLITK" if group_size >= 64 else "GEMM"
    if m <= 64:
        return "GEMM_SPLITK"
    return "GEMM"


def _gemlite_candidate_matmul_types(m: int, group_size: int) -> tuple[str, ...]:
    auto = _gemlite_auto_matmul_type(m, group_size)
    candidates = [
        auto,
        "GEMV_SPLITK",
        "GEMV_REVSPLITK",
        "GEMV",
        "GEMM_SPLITK",
        "GEMM",
    ]
    ordered = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return tuple(ordered)


def _make_case(
    *,
    m: int,
    k: int,
    n: int,
    mode: str,
    bits_arg: int,
    group_size_arg: int,
    axis: str,
    dtype: str,
    seed: int,
    native_layout: str,
):
    mode, group_size, bits, _ = resolve_qparams(mode, group_size_arg, bits_arg)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x_t = torch.randn((m, k), device="cuda", dtype=_torch_dtype(dtype))

    if axis == "col":
        code_shape = (n, k)
        meta_shape = (n, k // group_size)
    elif axis == "row":
        code_shape = (k, n)
        meta_shape = (k, n // group_size)
    else:
        raise ValueError(f"axis must be row or col, got {axis!r}")

    codes_t = torch.randint(0, 1 << bits, code_shape, device="cuda", dtype=torch.uint8)
    if mode == "affine":
        scales_t = (torch.rand(meta_shape, device="cuda", dtype=torch.float32) * 0.02 + 0.001).to(_torch_dtype(dtype))
        zeros_t = torch.randint(0, 1 << bits, meta_shape, device="cuda", dtype=torch.int16).to(_torch_dtype(dtype))
    elif mode == "nf4":
        scales_t = (torch.rand(meta_shape, device="cuda", dtype=torch.float32) * 0.02 + 0.001).to(_torch_dtype(dtype))
        zeros_t = None
    else:
        scales_t = torch.randint(0, 256, meta_shape, device="cuda", dtype=torch.uint8)
        zeros_t = None

    x_j = jax.dlpack.from_dlpack(x_t)
    codes_j = jax.dlpack.from_dlpack(codes_t).astype(jnp.uint32)
    scales_j = jax.dlpack.from_dlpack(scales_t)
    zeros_j = None if zeros_t is None else jax.dlpack.from_dlpack(zeros_t)
    wq_j = _pack_bits(codes_j, bits, strict_shape_alignment=bits in {1, 2, 4, 8}).block_until_ready()
    if native_layout == "kmajor":
        if axis != "col":
            raise ValueError("native_layout='kmajor' is only defined for axis='col'.")
        wq_j = jnp.swapaxes(wq_j, -2, -1).block_until_ready()
        scales_j = jnp.swapaxes(scales_j, -2, -1).block_until_ready()
        if zeros_j is not None:
            zeros_j = jnp.swapaxes(zeros_j, -2, -1).block_until_ready()
    return mode, group_size, bits, x_t, codes_t, scales_t, zeros_t, x_j, wq_j, scales_j, zeros_j


def _run_gemlite(
    *,
    x_t: torch.Tensor,
    codes_t: torch.Tensor,
    scales_t: torch.Tensor,
    zeros_t: torch.Tensor | None,
    bits: int,
    group_size: int,
    dtype: str,
    axis: str,
    mode: str,
    warmup: int,
    iters: int,
    matmul_type: str,
    inner_iters: int,
) -> Timing | str:
    if mode != "affine" or axis != "col" or bits not in {1, 2, 4, 8} or zeros_t is None:
        return "skip"
    from gemlite import GemLiteLinearTriton

    layer = GemLiteLinearTriton(
        W_nbits=bits,
        group_size=group_size,
        in_features=x_t.shape[1],
        out_features=codes_t.shape[0],
        input_dtype=_gemlite_dtype(dtype),
        output_dtype=_gemlite_dtype(dtype),
    )
    layer.pack(codes_t, scales_t, zeros_t, fma_mode=True, packing_bitwidth=32)
    torch.cuda.synchronize()
    if matmul_type == "auto":
        matmul_type = _gemlite_auto_matmul_type(x_t.shape[0], group_size)
    if matmul_type == "best":
        timings: list[Timing] = []
        errors: list[str] = []
        for candidate in _gemlite_candidate_matmul_types(x_t.shape[0], group_size):
            try:
                timings.append(
                    _time_torch(
                        lambda candidate=candidate: layer.forward_manual(x_t, matmul_type=candidate),
                        warmup,
                        iters,
                        inner_iters,
                    )
                )
            except Exception as exc:
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
        if timings:
            return min(timings, key=lambda timing: timing.mean_ms)
        return "error: no valid GemLite matmul_type; " + "; ".join(errors)
    try:
        return _time_torch(lambda: layer.forward_manual(x_t, matmul_type=matmul_type), warmup, iters, inner_iters)
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def _run_ejkernel(
    *,
    backend: str,
    x_j: jax.Array,
    wq_j: jax.Array,
    scales_j: jax.Array,
    zeros_j: jax.Array | None,
    mode: str,
    bits: int,
    group_size: int,
    axis: str,
    warmup: int,
    iters: int,
    inner_iters: int,
    block_n: int,
    block_k: int,
) -> Timing | str:
    try:
        cfg = None
        if block_n > 0 or block_k > 0:
            cfg = QuantizedMatmulConfig(
                block_n=block_n,
                block_k=block_k,
                platform=backend,
                backend="gpu",
            )
        if backend == "xla":
            if zeros_j is None:
                fn = jax.jit(
                    lambda a, b, s: quantized_matmul(
                        a,
                        b,
                        s,
                        None,
                        axis=axis,
                        mode=mode,
                        bits=bits,
                        group_size=group_size,
                        fuse=False,
                        platform="xla",
                    )
                )
                return _time_jax(lambda: fn(x_j, wq_j, scales_j), warmup, iters, inner_iters)
            fn = jax.jit(
                lambda a, b, s, z: quantized_matmul(
                    a,
                    b,
                    s,
                    z,
                    axis=axis,
                    mode=mode,
                    bits=bits,
                    group_size=group_size,
                    fuse=False,
                    platform="xla",
                )
            )
            return _time_jax(lambda: fn(x_j, wq_j, scales_j, zeros_j), warmup, iters, inner_iters)

        if zeros_j is None:
            fn = jax.jit(
                lambda a, b, s: quantized_matmul(
                    a,
                    b,
                    s,
                    None,
                    axis=axis,
                    mode=mode,
                    bits=bits,
                    group_size=group_size,
                    fuse=True,
                    strict_fuse=True,
                    allow_dense_fallback=False,
                    platform=backend,
                    cfg=cfg,
                )
            )
            return _time_jax(lambda: fn(x_j, wq_j, scales_j), warmup, iters, inner_iters)
        fn = jax.jit(
            lambda a, b, s, z: quantized_matmul(
                a,
                b,
                s,
                z,
                axis=axis,
                mode=mode,
                bits=bits,
                group_size=group_size,
                fuse=True,
                strict_fuse=True,
                allow_dense_fallback=False,
                platform=backend,
                cfg=cfg,
            )
        )
        return _time_jax(lambda: fn(x_j, wq_j, scales_j, zeros_j), warmup, iters, inner_iters)
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def _format_result(result: Timing | str) -> str:
    if isinstance(result, str):
        return result
    return f"{result.mean_ms:.4f} / {result.median_ms:.4f} / {result.min_ms:.4f} / {result.max_ms:.4f}"


def _speedup(gemlite: Timing | None, result: Timing | str) -> str:
    if gemlite is None or isinstance(result, str):
        return "-"
    return f"{gemlite.mean_ms / result.mean_ms:.3f}x"


def _csv_result_row(
    *,
    workload: str,
    backend: str,
    mode: str,
    axis: str,
    bits: int,
    group_size: int,
    inner_iters: int,
    result: Timing | str,
    gemlite: Timing | None,
) -> dict[str, str] | None:
    if isinstance(result, str):
        return None
    speedup = "" if backend == "gemlite" or gemlite is None else f"{gemlite.mean_ms / result.mean_ms:.9f}"
    return {
        "workload": workload,
        "backend": backend,
        "mode": mode,
        "axis": axis,
        "bits": str(bits),
        "group_size": str(group_size),
        "inner_iters": str(inner_iters),
        "mean_ms": f"{result.mean_ms:.9f}",
        "median_ms": f"{result.median_ms:.9f}",
        "min_ms": f"{result.min_ms:.9f}",
        "max_ms": f"{result.max_ms:.9f}",
        "speedup_vs_gemlite": speedup,
    }


def _write_csv(path: Path, rows: list[dict[str, str]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--bits", default="2", help="Comma list or 'all'.")
    parser.add_argument("--modes", default="affine")
    parser.add_argument("--axes", default="col")
    parser.add_argument("--group-size", default="128", help="Comma list of group sizes.")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--backends", default="gemlite,tilelang,cuda,triton,cute,xla")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--inner-iters",
        type=int,
        default=1,
        help="Time this many repeated launches per sample and report per-launch milliseconds.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native-layout", choices=("row", "kmajor"), default="row")
    parser.add_argument(
        "--gemlite-matmul-type",
        choices=("best", "auto", "GEMM", "GEMM_SPLITK", "GEMV", "GEMV_SPLITK", "GEMV_REVSPLITK"),
        default="best",
        help="GemLite matmul path. Use best to time all valid GemLite paths and keep the fastest.",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help="Require every non-GemLite backend with a GemLite baseline to reach this mean-speedup.",
    )
    parser.add_argument("--block-n", type=int, default=0, help="Optional native backend block_n override.")
    parser.add_argument("--block-k", type=int, default=0, help="Optional native backend block_k override.")
    parser.add_argument(
        "--workload-name",
        default=None,
        help="Optional workload label written to --csv-output.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Write successful timing rows to this CSV path.",
    )
    parser.add_argument(
        "--csv-append",
        action="store_true",
        help="Append rows to --csv-output instead of replacing it.",
    )
    args = parser.parse_args()

    backends = _csv(args.backends)
    failures: list[str] = []
    csv_rows: list[dict[str, str]] = []
    workload = args.workload_name or f"m{args.m}_k{args.k}_n{args.n}"
    print("backend | mode | axis | bits | group | mean/median/min/max ms | speedup_vs_gemlite")
    print("-|-|-|-|-|-|-")
    for mode in _csv(args.modes):
        for axis in _csv(args.axes):
            for group_size_arg in _int_csv(args.group_size):
                for bits_arg in _int_csv(args.bits):
                    case = _make_case(
                        m=args.m,
                        k=args.k,
                        n=args.n,
                        mode=mode,
                        bits_arg=bits_arg,
                        group_size_arg=group_size_arg,
                        axis=axis,
                        dtype=args.dtype,
                        seed=args.seed + bits_arg + group_size_arg,
                        native_layout=args.native_layout,
                    )
                    mode_n, group_size, bits, x_t, codes_t, scales_t, zeros_t, x_j, wq_j, scales_j, zeros_j = case
                    gemlite_baseline = None
                    if "gemlite" in backends:
                        baseline = _run_gemlite(
                            x_t=x_t,
                            codes_t=codes_t,
                            scales_t=scales_t,
                            zeros_t=zeros_t,
                            bits=bits,
                            group_size=group_size,
                            dtype=args.dtype,
                            axis=axis,
                            mode=mode_n,
                            warmup=args.warmup,
                            iters=args.iters,
                            matmul_type=args.gemlite_matmul_type,
                            inner_iters=args.inner_iters,
                        )
                        if isinstance(baseline, Timing):
                            gemlite_baseline = baseline
                        print(f"gemlite | {mode_n} | {axis} | {bits} | {group_size} | {_format_result(baseline)} | -")
                        gc.collect()
                        torch.cuda.empty_cache()
                        row = _csv_result_row(
                            workload=workload,
                            backend="gemlite",
                            mode=mode_n,
                            axis=axis,
                            bits=bits,
                            group_size=group_size,
                            inner_iters=args.inner_iters,
                            result=baseline,
                            gemlite=gemlite_baseline,
                        )
                        if row is not None:
                            csv_rows.append(row)
                    for backend in backends:
                        if backend == "gemlite":
                            continue
                        else:
                            result = _run_ejkernel(
                                backend=backend,
                                x_j=x_j,
                                wq_j=wq_j,
                                scales_j=scales_j,
                                zeros_j=zeros_j,
                                mode=mode_n,
                                bits=bits,
                                group_size=group_size,
                                axis=axis,
                                warmup=args.warmup,
                                iters=args.iters,
                                inner_iters=args.inner_iters,
                                block_n=args.block_n,
                                block_k=args.block_k,
                            )
                        speedup = _speedup(gemlite_baseline, result)
                        result_text = _format_result(result)
                        print(f"{backend} | {mode_n} | {axis} | {bits} | {group_size} | {result_text} | {speedup}")
                        row = _csv_result_row(
                            workload=workload,
                            backend=backend,
                            mode=mode_n,
                            axis=axis,
                            bits=bits,
                            group_size=group_size,
                            inner_iters=args.inner_iters,
                            result=result,
                            gemlite=gemlite_baseline,
                        )
                        if row is not None:
                            csv_rows.append(row)
                        if args.min_speedup is not None and gemlite_baseline is not None and isinstance(result, Timing):
                            value = gemlite_baseline.mean_ms / result.mean_ms
                            if value < args.min_speedup:
                                failures.append(
                                    f"{backend} {mode_n} axis={axis} bits={bits} group={group_size}: "
                                    f"{value:.3f}x < {args.min_speedup:.3f}x"
                                )
    if args.csv_output is not None:
        _write_csv(args.csv_output, csv_rows, args.csv_append)
        print(f"wrote {len(csv_rows)} rows to {args.csv_output}")
    if failures:
        raise SystemExit("speedup threshold failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
