# Multi-host OpenAI serving with eSurge

`easydel.inference.esurge.serving` serves one eSurge engine over the OpenAI-compatible API across every host of a
TPU slice, from one YAML file. `esurge-dashboard` (Rust, `libs/esurge-dashboard`) shows live engine metrics.

## Pieces

| piece | where | job |
|---|---|---|
| config | `serving/config.py` | typed, validated deployment YAML; unknown keys are errors |
| keys | `serving/keys.py` | the stable API key file |
| cluster | `serving/cluster.py` | bind addresses, per-launch coordination token |
| app | `serving/app.py`, `__main__.py` | per-host entry point: build engine, leader serves HTTP, workers replay |
| HTTP/OAI | `server/api_server.py` | existing `eSurgeApiServer` (unchanged API) |
| auth | `workers/esurge/auth/` | key manager, header-only extraction (`credentials.py`) |
| metrics | `esurge/metrics.py` | `MetricsCollector.model_summary()` behind `GET /metrics` |
| fan-out | `eray tpu job …` (eray) | start/stop/status/logs of one detached process per host |
| dashboard | `libs/esurge-dashboard` | server-rendered HTML over `GET /metrics` |

## Launch

Every host needs the same code, the same Python environment and the same YAML. From any host of the slice:

```bash
# validate (off-slice: ENABLE_DISTRIBUTED_INIT=0 keeps `import easydel` from joining a world)
ENABLE_DISTRIBUTED_INIT=0 python -m easydel.inference.esurge.serving check --config serve.yaml

# copy the tree to the other hosts, then start on all of them
eray tpu sync --ssh-key ~/.ssh/google_compute_engine ~/easydel ~/easydel --exclude .git --exclude target
eray tpu job start --name esurge --ssh-key ~/.ssh/google_compute_engine \
  -- ~/venv/bin/python -m easydel.inference.esurge.serving serve --config ~/serve.yaml
```

Hosts are discovered from the TPU VM's metadata (or `--tpu-name/--project/--zone`, or `--ips`). `start` checks
every host first and starts nothing unless all are free (no process holds `/dev/vfio/*`, job not already running).
Each process joins the JAX world through `import easydel`, which detects the slice from GCE metadata; nothing
numbers ranks by hand. The leader (`jax.process_index() == 0`) logs one line when serving:

```
[esurge-serving rank=0] READY model=… key_id=… key_file=… urls={'internal': 'http://10.x.x.x:8000/v1'}
```

## Config

See `examples/inference/glm53_flash_w8a8_serving.yaml`. Sections:

- `model`: `checkpoint` (gs:// works), `tokenizer` (a local path when the checkpoint is on GCS: Hugging Face
  tokenizers cannot read gs://), `served_name`, `sharding_axis_dims` in `(pp, dp, fsdp, ep, tp, sp)` order, dtypes,
  `attn_mechanism`, `quantization: {dtype, pattern}`; without `pattern` the model's
  `default_quantization_pattern()` is used.
- `esurge`: engine sections `runtime`, `cache`, `context`, `workers`, `parsing`, `vision`, validated by eSurge.
  `distributed` is not allowed; the `coordination` section owns it.
- `server`: `host` (`internal` = this VM's VPC address, the default), `port`, `api_key_file`, `log_level`,
  `max_concurrent_generations`.
- `coordination`: `control_port`, `bind_host` (`internal`), `heartbeat_timeout_s`, `token_timeout_s`.
- `compilation_cache_dir`: JAX persistent compilation cache.

**Pin the KV pool.** Without `cache.max_cache_tokens` the page pool is sized from free HBM, which differs by a few
hundred pages between launches. The page count is part of every compiled program's shape, so an unpinned pool
misses the persistent compilation cache on every launch and the full warmup recompiles.

**Share the compilation cache.** JAX writes persistent-cache entries only from the process whose distributed
process id is 0. That is not necessarily the HTTP leader (on the slice this was measured on, the cache was written on
worker 0 while `jax.process_index() == 0` was worker 12). With a host-local `compilation_cache_dir`, every other host
misses and recompiles, and the slice runs at their pace. After the first full warmup, copy the one non-empty cache
directory to every host:

```bash
eray tpu sync --ssh-key … /dev/shm/esurge/jax_cc /dev/shm/esurge/jax_cc   # run on the host that holds the entries
```

## Auth

- One API key, stored in `server.api_key_file`. It is created (mode 0600) on first launch and reused on every later
  one, so clients and the dashboard never need updating. Rotate it by replacing the file and restarting. A file
  readable by other users, empty, or shorter than 32 characters stops the server from starting.
- Keys are accepted only in `Authorization: Bearer <key>` or `X-API-Key: <key>`. Query strings and request-body
  fields (including OpenAI's `user`) are never read as credentials.
- Fail closed: with `require_api_key=True` (always, for this deployment) a missing auth manager or an unknown key
  gives 401, never open access. Permission failures on a valid key give 403, rate/quota limits 429.
- Key lookup hashes first and confirms the digest with `hmac.compare_digest`; no structure is keyed by a raw key.
- The ZMQ coordination token is unrelated to the API key. The leader generates a fresh one per launch and hands
  it to the workers through the JAX coordination service; it is never written to disk. The control plane binds the
  leader's internal address only.

## Usage fields and `/metrics`

Every completion's `usage` carries `first_token_time` (TTFT), `tokens_per_second`, `cache_hit`, `cache_read`
(prompt tokens from the prefix cache), `uncached_input` (prompt tokens prefilled) and `cache_written` (whole prompt
pages newly committed to the prefix cache).

`GET /metrics` (same key) adds `engine_metrics.<model>`: running/waiting requests, a 300-second window with request
and token cache-hit rates, TTFT and latency p50/p99, prefill tok/s (uncached input over TTFT, including queueing),
decode tok/s (tokens after the first over time after the first), output tok/s (all completions over the window's
span), and the 20 most recent requests.

## Dashboard

```bash
cd libs/esurge-dashboard && cargo build --release
target/release/esurge-dashboard --upstream http://<leader-internal-ip>:8000 --listen <internal-ip>:9090
```

It has no key of its own: each request's key (Bearer, `X-API-Key`, or a browser login at `/login`, kept server-side
behind an HttpOnly SameSite=Strict session cookie) is forwarded to the API's `/metrics`, which decides. Routes: `/`,
`/api/metrics`, `/login`, `/logout`, `/healthz`.

## Operations

```bash
eray tpu job status --name esurge --ssh-key …        # running / exited (rc) / stopped, per host
eray tpu job logs   --name esurge --ssh-key … --host 0 -n 100
eray tpu job stop   --name esurge --ssh-key …        # SIGTERM the process group, SIGKILL after --grace (30 s),
                                                    # wait up to --kill-wait (30 s), then report TPU holders
```

Logs live on each host in `~/.eray/hostjobs/<name>/log` (previous run in `log.1`). Restart = `stop`, then `start`.

`stop` signals every host at once. `serve()` turns SIGTERM into a clean exit: the leader finishes uvicorn's graceful
HTTP shutdown and then closes the engine, which sends Shutdown to the workers. A worker that gets SIGTERM waits up to
20 s for that Shutdown before closing on its own. Each host logs `[esurge-serving] stopped by SIGTERM`. A process
busy inside an XLA compile cannot run the Python handler until the compile returns, so a stop during warmup can
fall through to SIGKILL; that is expected.

`eray tpu sync SRC DEST` copies to every host, including this one when `DEST` is a different path from `SRC`.

## Measured on a v5p-128 (64 chips, GLM-5.3-Flash W8A8, 4 DP × EP4×TP4)

| | |
|---|---|
| cold start (empty compile cache, 28 model + 8 sampler variants) | 77 min |
| warm start (pinned pool, process-0 cache copied to every host) | 17 min |
| clean stop of all 16 hosts by SIGTERM | ~70 s |

Closed-loop streaming chat, 512 output tokens, ~940-token prompts sharing a system preamble (95 % of prompt tokens
from the prefix cache), 180 s per level, no errors:

| concurrency | output tok/s | TTFT p50 / p99 | latency p50 / p99 |
|---|---|---|---|
| 1 | 50 | 0.07 / 0.58 s | 10.2 / 10.7 s |
| 32 | 1,129 | 0.25 / 0.51 s | 14.5 / 14.8 s |
| 128 | 1,705 | 0.33 / 2.7 s | 32.4 / 49.9 s |

DP routing sends a request to the worker that already holds its prompt prefix while that worker has a free row,
otherwise to the least-loaded worker with room. A shared system prompt therefore fills one DP worker before spreading
(at concurrency 32 all requests ran on one worker); at 128 every worker is busy.

## Known limits

- One model per deployment; one uvicorn worker (generation runs in the engine, not the HTTP workers).
- A world is all or nothing: losing any host aborts it, and a restart is a full relaunch.
- Full warmup compiles every token/request bucket; it is fast only when the compilation cache matches, which needs
  a pinned KV pool, the same code and config, and the process-0 cache present on every host (see Config).
- A shared prompt prefix concentrates moderate load on one DP worker (see the routing note above); latency at mid
  concurrency could improve with routing that also weighs per-worker load below the row cap.
- `/health` requires the API key like every other route.
- Workarounds that exist only in the batch data-generation harness (an explicit KV budget, EOS seeding) are not part
  of this serving path.
- `external` URLs are reachable only if a firewall rule allows the port.
- Prefill tok/s in `/metrics` includes queueing time, so it is a lower bound on raw prefill speed.
