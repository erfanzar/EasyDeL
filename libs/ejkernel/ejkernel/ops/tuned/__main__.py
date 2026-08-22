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

"""CLI for the tuned-kernel table.

    python -m ejkernel.ops.tuned stats
    python -m ejkernel.ops.tuned dump --kernel grouped_matmul
    python -m ejkernel.ops.tuned sweep --kernel grouped_matmul --out /tmp/gmm.db
    python -m ejkernel.ops.tuned merge --into <shipped.db> /tmp/gmm.db

``dump`` exists because the table is a binary file in the repository: a sweep's
effect should be reviewable as text, and a regression should be diffable,
without anyone opening SQLite by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from ._store import TunedStore, default_db_path, merge, open_for_write, upsert
from ._sweep import run_sweep, sweep_specs


def _load_sweeps() -> None:
    """Import sweep modules so their specs register."""
    from ._sweeps import grouped_matmul  # noqa: F401

    try:
        from ._sweeps import from_registry  # noqa: F401
    except Exception as exc:  # registry-driven sweeps need a source checkout
        print(f"(registry-driven sweeps unavailable: {exc})", file=sys.stderr)


def cmd_stats(args) -> int:
    store = TunedStore(args.db) if args.db else TunedStore()
    if not store.available():
        print(f"no tuned table at {store.path}")
        return 1
    entries = store.entries()
    print(f"{store.path}  ({len(entries)} rows)")
    by_kernel: dict[str, list] = {}
    for e in entries:
        by_kernel.setdefault(e.kernel, []).append(e)
    for kernel, rows in sorted(by_kernel.items()):
        devices = sorted({r.device for r in rows})
        platforms = sorted({r.platform for r in rows})
        timed = [r for r in rows if r.ms]
        speedups = [s for s in (r.speedup_over_runner_up() for r in rows) if s]
        print(f"  {kernel}: {len(rows)} rows | devices={devices} | platforms={platforms}")
        if timed:
            print(f"      timed={len(timed)}")
        if speedups:
            print(
                f"      speedup over runner-up: median {sorted(speedups)[len(speedups) // 2]:.2f}x "
                f"max {max(speedups):.2f}x"
            )
    return 0


def cmd_dump(args) -> int:
    store = TunedStore(args.db) if args.db else TunedStore()
    entries = store.entries(args.kernel)
    if not entries:
        print("no rows", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "kernel": e.kernel,
                        "device": e.device,
                        "dtypes": e.dtypes,
                        "shape": e.shape_key,
                        "platform": e.platform,
                        "config": e.config,
                        "ms": e.ms,
                        "runner_up": e.runner_up,
                    }
                    for e in entries
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    for e in entries:
        ms = f"{e.ms:.4f}ms" if e.ms else "-"
        gain = e.speedup_over_runner_up()
        gain_s = f" ({gain:.2f}x over {e.runner_up['platform']})" if gain else ""
        cfg = json.dumps(e.config, sort_keys=True)
        print(f"{e.kernel}\t{e.device}\t{e.dtypes}\t{e.shape_key}\t{e.platform}\t{cfg}\t{ms}{gain_s}")
    return 0


def cmd_sweep(args) -> int:
    _load_sweeps()
    specs = sweep_specs()
    if args.list or not args.kernel:
        for name, spec in sorted(specs.items()):
            print(f"{name}: {spec.description}")
        return 0
    spec = specs.get(args.kernel)
    if spec is None:
        print(f"unknown kernel {args.kernel!r}; known: {sorted(specs)}", file=sys.stderr)
        return 2

    extra = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        try:
            extra[key] = json.loads(value)
        except json.JSONDecodeError:
            extra[key] = value

    import ejkernel

    provenance = {
        "sweep": spec.kernel,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "ejkernel": getattr(ejkernel, "__version__", "unknown"),
        "options": extra or None,
    }

    def progress(label, entry):
        over_default = entry.speedup_over_default()
        gain = entry.speedup_over_runner_up()
        bits = []
        if over_default:
            bits.append(f"{over_default:.2f}x over default {entry.baseline['config']}")
        if gain:
            bits.append(f"{gain:.2f}x over 2nd")
        tail = ("  (" + ", ".join(bits) + ")") if bits else ""
        print(f"  {label}\n      -> {entry.platform} {entry.config}  {entry.ms:.4f}ms{tail}", flush=True)

    print(f"sweeping {spec.kernel} ...", flush=True)
    entries = run_sweep(
        spec, reps=args.reps, min_gain=args.min_gain, provenance=provenance, on_point=progress, **extra
    )
    if not entries:
        print("no measurable points", file=sys.stderr)
        return 1
    out = args.out or str(default_db_path())
    conn = open_for_write(out)
    written = upsert(conn, entries)
    conn.close()
    print(f"wrote {written} rows -> {out}")
    return 0


def cmd_merge(args) -> int:
    written = merge(args.into, *args.sources, prefer_faster=not args.last_wins)
    print(f"merged {written} rows into {args.into}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ejkernel.ops.tuned", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="summarize a tuned table")
    p.add_argument("--db")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("dump", help="print rows as text (reviewable/diffable)")
    p.add_argument("--db")
    p.add_argument("--kernel")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("sweep", help="measure a kernel and write rows")
    p.add_argument("--kernel")
    p.add_argument("--out", help="target db (default: the shipped table)")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument(
        "--min-gain",
        type=float,
        default=1.05,
        help="minimum speedup over the untuned default before a row is written (default 1.05)",
    )
    p.add_argument("--list", action="store_true", help="list registered sweeps")
    p.add_argument("--set", action="append", metavar="KEY=JSON", help="sweep option, repeatable")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("merge", help="merge tuned tables")
    p.add_argument("--into", required=True)
    p.add_argument("sources", nargs="+")
    p.add_argument("--last-wins", action="store_true", help="ignore timings when resolving conflicts")
    p.set_defaults(fn=cmd_merge)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
