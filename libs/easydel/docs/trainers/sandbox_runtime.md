# Sandbox Runtime

Sandbox-backed environments (see [RL environments](environments.md)) execute model output through a
`SandboxSession` obtained from a `KiriRuntime` or a `SandboxPool` (`easydel.environments.runtime`). Daemon
installation and the pod bridge live in `easydel.environments.kiri_installer`.

## Backends

| `backend`          | behaviour                                                                                              |
|--------------------|--------------------------------------------------------------------------------------------------------|
| `"kiri"` (default) | reuse a running Kiri daemon or boot an **installed** `kiri` binary; never downloads unless `auto_install=True` |
| `"docker"`         | require a Docker-compatible daemon (only `DOCKER_HOST` is consulted)                                   |
| `"auto"`           | try Kiri, then Docker, then the local backend; a local fallback forces `use_jail="required"`            |
| `"local"`          | child processes in a persistent per-episode temporary workspace (`LocalSandboxSession`)                |

Container backends produce a `ContainerSandboxSession` (`jailed` is always `True`). `discover_engine_endpoint(backend)`
returns `(endpoint, resolved_backend)` without creating a session.

```python
from easydel.environments import ContainerConfig, KiriRuntime, SandboxPool

with KiriRuntime(backend="local", config=ContainerConfig(use_jail="required")) as rt:
    session = rt.create_session()
    print(session.run_bash("echo hi").stdout)
    session.close()

pool = SandboxPool(["shm:4"])              # boot 4 RAM-backed daemons in /dev/shm
session = pool.acquire(affinity_key="rollout-17")
```

`SandboxPool(endpoints=None, *, backend="kiri", config=None, token=None, auto_install=False)` routes
`acquire(affinity_key=...)` by SHA-256 of the key (round robin when the key is `None`), so every turn of one
trajectory lands on the same daemon. Endpoint specs `"shm:N"` / `"pod-shm:N"` boot local daemons;
`SandboxPool.from_local_shm(num_daemons=4, ...)` and `SandboxPool.from_pod_shm(num_daemons_per_host=8, ...)` are the
explicit constructors.

## `ContainerConfig`

| field                                       | default             | notes                                                                    |
|---------------------------------------------|---------------------|--------------------------------------------------------------------------|
| `image`, `workdir`                          | `python:3.11-slim`, `/workspace` | container backends                                          |
| `memory_mb`, `cpus`, `pids_limit`           | `2048`, `1.0`, `256` |                                                                         |
| `network`                                   | `None`              | `None` disables network where enforceable; explicit `False` demands isolation |
| `read_only_rootfs`                          | `False`             |                                                                          |
| `tmpfs_mb`, `workspace_tmpfs_mb`            | `256`, `0`          |                                                                          |
| `timeout`                                   | `30.0`              | default per-command seconds                                              |
| `max_output_chars`                          | `16384`             | per output stream                                                        |
| `max_stdin_bytes`                           | 16 MiB              | larger stdin is rejected with a failed `ExecResult`                      |
| `max_file_size_mb`                          | `1024`              | `RLIMIT_FSIZE` for local commands (`0` disables)                         |
| `use_jail`                                  | `"auto"`            | local backend jail policy, see below                                     |

### `use_jail` (local backend)

| value        | behaviour                                                                                                     |
|--------------|---------------------------------------------------------------------------------------------------------------|
| `"auto"`     | jail every command with `bwrap`; raise `SandboxInfraError` if `bwrap` is not on `PATH`                        |
| `"required"` | same as `"auto"`                                                                                              |
| `"off"`      | run model commands **unjailed**; raises if `network=False` or `read_only_rootfs=True` (cannot be enforced)     |

Only `"off"` runs unjailed. `swe_repo`, `terminal` and `harness` default to `ContainerConfig(use_jail="required")`.

## Exec deadlines and output caps

- **Container exec**: `timeout` is one monotonic deadline for the whole call. On expiry the command's process group is
  SIGKILLed inside the container (via a recorded pid file) and the result has `timed_out=True`; an in-container
  SIGKILL backstop fires about two seconds after the client deadline. Output is read up to a byte cap derived from
  `max_output_chars`; a command exceeding it is killed and reported as failed with partial output.
- **Local exec**: each stream keeps at most `max(4 * max_output_chars, 4096)` bytes; every command leads its own
  process group.
- `ExecResult.combined_output(max_chars)` truncates with an `... (output truncated)` marker.
- `kill_sandbox_processes()` runs before grading in `swe_repo` / `terminal`: containers run `kill -9 -1` as root
  (pid 1 is spared); local sessions SIGKILL every process group they spawned.

## Kiri daemons

### Installation (opt-in)

Nothing is downloaded implicitly. Install a pinned, SHA-256-verified release with:

```bash
easydel setup kiri [--version v0.2.1] [--install-dir DIR] [--force] [--start] [--data-dir /dev/shm/kiri] [--num-daemons N]
easydel-install-kiri [--version ...] [--install-dir DIR] [--force] [--start]
```

or pass `auto_install=True` to `KiriRuntime` / `SandboxPool`. Daemons started by the CLI with `--start` are
persistent.

### Ownership and shutdown

Daemons booted by this process are registered as owned (`owned_kiri_daemons()`) and stopped by
`KiriRuntime.close()`, `SandboxPool.close()` (both also context managers), `shutdown_kiri_daemons(endpoints=None,
grace=2.0)` (SIGTERM to the daemon's process group, SIGKILL after `grace`), and at interpreter exit (`atexit`).
Daemons started by other processes, or with `persist=True`, are never signalled; a runtime that merely reused a daemon
leaves it running.

### `KiriDaemonBudget`

Per-daemon caps for RAM-backed pools; every container created on that daemon is clamped by `clamp_host_config`.

| field        | default | caps                                                                 |
|--------------|---------|----------------------------------------------------------------------|
| `cpus`       | `4.0`   | `NanoCpus` per container                                             |
| `memory_mb`  | `8192`  | `Memory` per container                                               |
| `pids_limit` | `1024`  | `PidsLimit` per container                                            |
| `tmpfs_mb`   | `4096`  | each tmpfs mount, and the daemon's data-dir storage (checked before boot) |

Pass it as `budget=` to `SandboxPool.from_local_shm` / `from_pod_shm`.

## Pod bridge

`SandboxPool.from_pod_shm(num_daemons_per_host=8, *, data_dir="/dev/shm/kiri", base_port=2375, host_ips=None,
budget=None, expose_bridge=False, bridge_token=None, allowed_peers=None)`:

- `expose_bridge=False` (default): local `unix://` endpoints only; passing `host_ips` raises.
- `expose_bridge=True` requires `bridge_token` (identical on every host, at least 16 characters). Each daemon gets a
  TCP bridge on this host's private IP at `base_port + i`; peers default to the gathered pod IPs (or `allowed_peers`),
  and the returned endpoints span every host.
- The bridge binds loopback unless told otherwise; a non-loopback bind without a token or an `allowed_peers` list
  raises `ValueError`. Clients send the token as the `X-EasyDeL-Sandbox-Token` header (stripped before forwarding);
  peers outside the allow-list are rejected.
- Refused requests: container creates with `Privileged`, host `Binds`, `VolumesFrom`, non-local volume drivers,
  mounts other than `volume`/`tmpfs`, devices, `CapAdd`, host or `container:` namespace modes, or `unconfined`
  security options; volume creates with non-local drivers or `DriverOpts`; privileged execs; and the
  `/plugins`, `/swarm`, `/services`, `/secrets`, `/configs`, `/nodes`, `/session`, `/build` API prefixes.

## Remote CPU VMs

```bash
easydel setup cpu-vm HOST [HOST ...] [--user U] [--ssh-key KEY] [--port 2375] [--num-daemons 1] \
  [--data-dir DIR] [--kiri-version v0.2.1] [--dry-run]
```

Installs Kiri over SSH and opens one `ssh -fN -L 127.0.0.1:<port>:<remote socket>` tunnel per daemon; the printed
endpoints are `http://127.0.0.1:<port>` and nothing listens on remote TCP. The former `--expose-tcp` (and
`--bind-host`) flags were removed and are rejected.

## Security model and limits

Guaranteed by the code:

- Local commands are jailed with `bwrap` unless `use_jail="off"` is set explicitly; a missing `bwrap` is an error,
  including on `backend="auto"` fallback.
- Graders kill agent processes before verification and treat a failed kill as unscorable.
- Bridges never listen on a routable address without a token and a peer allow-list, and refuse the sandbox-escaping
  requests listed above.
- Only daemons this process started are ever stopped.

Not guaranteed:

- Unjailed local mode (`use_jail="off"`) only kills the process groups the session spawned; a process that called
  `setsid()` escapes cleanup. Treat it as trusted-code-only.
- `kill_sandbox_processes` on container sessions (`kill -9 -1` exec) is covered only against a fake engine in the
  test suite, not against a real Kiri or Docker daemon.
- The bridge policy filters the Docker API surface; isolation of permitted containers is whatever the engine
  provides.
- `swe_repo` restores a fixed list of test-infrastructure file names; other ways a repository can hook its own test
  run are not covered.
