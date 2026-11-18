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
- `compilation_cache_dir`: JAX persistent compilation cache. Optional. Resolution order (also used by a plain
  `eSurge(...)` in library code, `easydel.utils.compilation_cache`):
  1. this setting, when given;
  2. a directory JAX already has (`JAX_COMPILATION_CACHE_DIR`);
  3. on Google Cloud, `gs://easydel-caching-<region>-<project number>/jax-<jax version>/<model>`, with the bucket
     created on first use in the VM's project and region (uniform access, public access prevented, 90-day object
     lifetime). A bucket of that name owned by another project is refused, because JAX runs the executables it
     finds in its cache. If the bucket cannot be created or read (e.g. the service account lacks
     `storage.buckets.create`), a warning is logged and the engine compiles without a shared cache;
  4. otherwise no persistent cache.

**Pin the KV pool.** Without `cache.max_cache_tokens` the page pool is sized from free HBM, which differs by a few
hundred pages between launches. The page count is part of every compiled program's shape, so an unpinned pool
misses the persistent compilation cache on every launch and the full warmup recompiles.

**Share the compilation cache.** JAX writes persistent-cache entries only from the process whose distributed
process id is 0. That is not necessarily the HTTP leader (on the slice this was measured on, the cache was written on
worker 0 while `jax.process_index() == 0` was worker 12). With a host-local `compilation_cache_dir`, every other host
misses and recompiles, and the slice runs at their pace. Point `compilation_cache_dir` at one shared directory
instead (for example `gs://<bucket>/jax_cc/<deployment>`; JAX reads and writes it through `etils[epath]`): process 0
writes each entry once and every host reads it, so the next launch is warm everywhere with no copy step. Keep one
directory per model, mesh and code version. A cache that has only ever lived on one host can be seeded once with
`gsutil -m cp -n <local_dir>/*-cache gs://<bucket>/jax_cc/<deployment>/`.

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

## Data-parallel replicas (dp > 1)

`dp: N` in `model.sharding_axis_dims` runs N independent engines behind one endpoint. Nothing else changes: same
YAML sections, same `serving serve --config X.yaml` on every host, no extra keys, flags or environment variables.
See `examples/inference/dp_replicas_serving.yaml`.

How it starts:

1. Every process joins the whole slice and builds the deployment's global mesh.
2. If every `dp` slice of that mesh is a contiguous box of whole hosts (true for the usual layouts on a TPU pod
   slice), each process works out its replica and its place in it, publishes its address through the slice's
   key-value store, leaves the slice world and re-executes the same command inside its replica's own JAX world.
   Otherwise (several dp slices on one host, non-TPU backends, `pp > 1`) the launch keeps one rank-major engine.
3. Each replica is a complete `dp = 1` engine: its own weights, KV and prefix cache, scheduler and step loop. Its
   JAX coordination service listens on `coordination.control_port + 1`, its request plane on `control_port`.
4. Launch-wide limits (`runtime.max_num_seqs`, `max_num_batched_tokens`, `max_num_seq_buckets`,
   `cache.max_cache_tokens`) are split evenly over the replicas, so the same config serves the same totals.
5. Replica 0's leader starts the **front**: the same command as a host-only process (JAX on CPU, no distributed
   world) that serves the OpenAI endpoint. It goes READY as soon as one replica answers and attaches the others as
   they come up. If the front exits, replica 0's leader restarts it with backoff.

Routing: prefix affinity sends a request to the replica that holds its prompt prefix while that replica has at most
4 more in-flight requests than the least-loaded one; beyond that it joins the shortest queue. A shared system prompt
therefore no longer concentrates load on one replica. Requests past the replicas' rows queue; they are not rejected.

Failures: a replica fails alone. When one of its hosts dies, its leader's step bookkeeping notices the missed ack
after `step_timeout_s` (30 s), tells the front, and the front ends that replica's in-flight requests with a 500 and
routes everything new to the remaining replicas. A failed replica refuses new router connections until its process
exits. It is not restarted automatically: bring it back with a relaunch.

Measured on a v5p-128 (GLM-5.3-Flash W8A8, `[1, 2, 1, 4, 8, 1]`, two replicas of 8 hosts):

| | dp = 1 engine on 32 chips | replica 0 | replica 1 |
|---|---|---|---|
| model step, 16 tokens | 14.98 ms | 14.88 ms | 15.22 ms |
| model step, 128 tokens | 25.64 ms | 25.57 ms | 25.83 ms |

| | |
|---|---|
| warm start, both replicas reading the shared compile cache | ~24 min |
| 256 concurrent 4K-token chats, one wave | 5,079-5,456 output tok/s (83-89 % of 2x one engine) |
| replica-0 host killed under 16 parallel streams | front stayed up; requests in flight on replica 0 got a 500 at +31 s; nothing hung; replica 1 kept serving |

For comparison, one rank-major engine over the same 64 chips stepped 21.6 ms at 16 tokens and 34.7 ms at 128,
because every step ran cross-rank collectives over both dp halves.

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
| warm start (pinned pool, process-0 cache readable by every host) | 17 min |
| clean stop of all 16 hosts by SIGTERM | ~70 s |

Closed-loop streaming chat, 512 output tokens, ~940-token prompts sharing a system preamble (95 % of prompt tokens
from the prefix cache), 180 s per level, no errors:

| concurrency | output tok/s | TTFT p50 / p99 | latency p50 / p99 |
|---|---|---|---|
| 1 | 50 | 0.07 / 0.58 s | 10.2 / 10.7 s |
| 32 | 1,129 | 0.25 / 0.51 s | 14.5 / 14.8 s |
| 128 | 1,705 | 0.33 / 2.7 s | 32.4 / 49.9 s |

Long context: one wave of ~83K-token prompts (unique per request, no prefix-cache hits) with 8,192 forced output
tokens (`ignore_eos`), streamed:

| concurrency | output tok/s | TTFT p50 / max | decode tok/s per request (p50 / min) | longest stream stall | latency |
|---|---|---|---|---|---|
| 1 | 43 | 10.8 / 10.8 s | 45.5 / 45.5 | 0.07 s | 191 s |
| 8 | 288 | 19.7 / 19.7 s | 39.4 / 38.6 | 3.8 s | 228 s |
| 32 | 835 | 50.6 / 81.2 s | 31.1 / 27.4 | 3.9 s | 314 s |

Prefill and decode share every step: while a long prompt prefills, every running stream on the slice pauses (the
3.8 s stalls), and TTFT at 32 is bounded by about 33K prefill tokens/s across the slice.

DP routing sends a request to the worker that already holds its prompt prefix while that worker has a free row,
otherwise to the least-loaded worker with room. A shared system prompt therefore fills one DP worker before spreading
(at concurrency 32 all requests ran on one worker); at 128 every worker is busy.

## Known limits

- One model per deployment; one uvicorn worker (generation runs in the engine, not the HTTP workers).
- A world is all or nothing: losing any host aborts it, and a restart is a full relaunch. With dp replicas the world
  is one replica: the others keep serving, but the failed replica comes back only with a relaunch.
- Full warmup compiles every token/request bucket; it is fast only when the compilation cache matches, which needs
  a pinned KV pool, the same code and config, and a `compilation_cache_dir` every host can read (see Config).
- With rank-major DP (one engine, `cache.data_parallelism_axis: dp`), a shared prompt prefix concentrates moderate
  load on one DP worker (see the routing note above). dp replicas use bounded affinity instead.
- `/health` requires the API key like every other route.
- Workarounds that exist only in the batch data-generation harness (an explicit KV budget, EOS seeding) are not part
  of this serving path.
- `external` URLs are reachable only if a firewall rule allows the port.
- Prefill tok/s in `/metrics` includes queueing time, so it is a lower bound on raw prefill speed.
