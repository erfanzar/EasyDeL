# eray job CLI

Design + implementation notes. Owner: erfanzar. Started 2026-07-03 (designed
after the 2026-07-02 ops session; every feature maps to a real incident).

## Philosophy

One tool for the operator loop — launch → watch → diagnose → stop → clean —
where every command tells the truth (Ray's job status lies: a remote raise
behind `print_remote_raise` still reports SUCCEEDED), every command takes
`--json`, and address resolution is zero-config on the pod
(`--address` → `RAY_ADDRESS` → `http://127.0.0.1:8265`).

## Command tree

```
eray
├── run       submit (env-inheriting, working-dir-defaulted)     [P1 ✅]
├── status    last N runs, truthful verdicts (alias: ps)         [P1 ✅]
├── logs      driver logs: follow / errors-only / grep           [P1 ✅]
├── stop      stop by id / --last                                [P1 ✅]
├── watch     live watcher: phases, --until-step, --alert       [✅]
├── doctor    disk, raylet logs, TPU locks, pkgs, nodes, --json  [✅]
├── clean     raylet | packages | sessions | all                 [✅]
├── nodes     alive nodes + TPU chip totals                      [✅]
├── rerun     resubmit recorded entrypoint+env                   [✅]
├── diff      git/env/entrypoint delta between two jobs          [✅]
└── tpu       connect/disconnect/… + bounce (--yes-kill-jobs)    [✅]

Everything above is implemented and unit-tested; status --watch refreshes,
logs --metrics renders a per-step metric table, run/--queue waits for idle,
run/doctor take --json. All log-scanning patterns are config-driven
(~/.eray/patterns.json / $ERAY_PATTERNS / ./.eray-patterns.json).
Deliberately dropped: watch --notify (headless pods have no notification
channel; exit codes + wandb serve that role).
```

## P1 behaviors (implemented in `eray/cli/jobs.py`)

### eray run

`eray run [opts] -- <entrypoint...>`

- `--working-dir .` is the default; `--working-dir PATH` replaces;
  `--no-working-dir` opts out.
- Package-size guard: warn > 500 MB, abort > 2 GB unless `--force-package`
  (the 9.8 GB home-dir packaging incident). Walk skips `.git`, `.venv`,
  `__pycache__`, cache dirs — approximating Ray's default excludes.
- `--env-inherit` (default on): current shell env is injected into the job's
  runtime env minus a deny-list of host-machine vars (`PATH`, `HOME`, `PWD`,
  `SHELL`, `TERM`, `USER`, `SSH_*`, `XDG_*`, `LC_*`, `VIRTUAL_ENV`,
  `CONDA_*`, `LD_*`, `RAY_*`, `TMPDIR`, …). `PYTHONPATH` passes through
  (load-bearing on pods with a shared fs layout). `--env K=V` overrides,
  `--env-file` supported, `--no-env-inherit` opts out. Values whose keys look
  secret (`TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL`) are masked in terminal echo.
- Submission id default: `<script-stem>-<user>-<YYYYmmdd-HHMMSS>`; `--id`
  overrides.
- Job metadata records git sha + dirty flag + cwd + user so `status` can
  answer "which code was that run?" (editor-clobber / stale-snapshot class
  of debugging). A local history line is appended to `~/.eray/history.jsonl`.
- `-f/--follow` tails driver logs after submit; default prints id, dashboard,
  and the copy-paste `eray logs` line, then returns.

### eray status (alias: ps)

Last 10 jobs (newest first; `-n` for more): ID, STATE, VERDICT, PHASE, AGE,
RUNTIME, ENTRYPOINT.

- VERDICT scans the driver log tail (filesystem fast-path via the Ray
  session logs dir when on the head; `get_job_logs` HTTP fallback) for error
  signatures — `Traceback`, `RESOURCE_EXHAUSTED`, `CompileTimeHbmOom`,
  runtime-env merge conflicts, abstract-leaf load failures. A SUCCEEDED job
  with a remote raise shows `failed(remote)`; exit code is non-zero when any
  displayed job has a failing verdict (CI-friendly).
- PHASE reads progress markers: packaging → loading → compiling →
  `step N (kl X)` — extracted from the same tail.

### eray logs

`eray logs [id|last]` — full driver log to stdout with progress-bar spam
filtered (`--raw` disables). `--errors` prints only tracebacks + signature
lines; `--grep PATTERN` filters; `-f/--follow` streams.

### eray stop

`eray stop <id>` or `eray stop --last`. No bare stop-everything.

## P2/P3 sketches

- `watch`: `--until-step N`, `--alert 'kl_loss>5'`, `--alert 'step_time>120'`,
  phase-transition + first-metrics + error-signature events.
- `doctor`: df per mount, raylet/gcs log sizes (>5 GB flagged, suggests
  `eray clean raylet`), alive nodes vs expected, dashboard reachability,
  TPU device lock holders, stale `_ray_pkg` bytes.
- `clean raylet` → `eray.core.monitoring.sweep_raylet_logs`;
  `clean packages` → `_ray_pkg_*` not referenced by RUNNING jobs
  (`--dry-run` default).
- `tpu bounce`: cluster-wide ray stop/start with the recorded head/worker
  resources + node-count verification, gated behind `--yes-kill-jobs`.
- Brainstormed: `rerun <id>` (resubmit recorded entrypoint+env),
  `diff <id1> <id2>` (git sha + env delta), `run --queue` (wait for idle).
