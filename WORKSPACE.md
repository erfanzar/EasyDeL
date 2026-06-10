# EasyDeL Stack — workspace guide

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
monorepo containing four packages that are developed together and released
independently:

| package  | path            | import     | PyPI           | mirror repo          |
| -------- | --------------- | ---------- | -------------- | -------------------- |
| EasyDeL  | `libs/easydel`  | `easydel`  | `easydel`      | (this repo)          |
| Spectrax | `libs/spectrax` | `spectrax` | `spectrax-lib` | `erfanzar/Spectrax`  |
| eJKernel | `libs/ejkernel` | `ejkernel` | `ejkernel`     | `erfanzar/ejkernel`  |
| eFormer  | `libs/eformer`  | `eformer`  | `eformer`      | `erfanzar/eformer`   |

**This monorepo is the source of truth.** The standalone repositories are
read-only mirrors kept in sync automatically — do not merge changes there
directly (a direct push makes the sync fail until reconciled with
`scripts/subtree-sync.sh pull <lib>`).

## Development

```bash
uv sync                              # all four packages editable in .venv
uv sync --group tpu                  # + TPU extras        (jax[tpu], ejkernel[tpu])
uv sync --group cuda                 # + CUDA extras
uv sync --group torch --group profile
```

During development the packages resolve against each other from `libs/`
(`{ workspace = true }` sources); published wheels keep the pinned PyPI
versions declared in `libs/easydel/pyproject.toml`.

## Layering contract (CI-enforced)

```
spectrax    ejkernel    eformer      <- independent of each other and of easydel
       \        |        /
              easydel                <- the only package that may import the others
```

`uv run lint-imports` checks this locally; Workspace CI runs it on every PR.

## Testing

Workspace CI runs affected-only *smoke* checks (imports + a tiny easydel
forward) plus the layering contract. The deep suites are hardware-bound and
run locally:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests -m "not slow"
uv run pytest libs/spectrax/tests
uv run pytest libs/eformer/tests
uv run pytest libs/ejkernel/test        # kernels: most need GPU/TPU
```

## Releasing

One command per package — bumps the version, keeps easydel's pins on the
sibling libraries in lockstep, refreshes the lock, commits, and tags:

```bash
scripts/release.sh ejkernel 0.0.81
git push origin HEAD ejkernel-v0.0.81   # tag push triggers the PyPI publish
```

`.github/workflows/publish.yaml` builds and publishes exactly the tagged
package (PyPI trusted publishing, or the `PYPI_API_TOKEN` secret).

## Mirror sync

`.github/workflows/sync-subtrees.yaml` re-splits `libs/{spectrax,ejkernel,eformer}`
on every push to `main`/`vnext` and fast-forwards the standalone repos
(requires the `SUBTREE_SYNC_TOKEN` secret: fine-grained PAT, Contents
read/write on the three mirrors). Manual fallback:

```bash
scripts/subtree-sync.sh push            # all three
scripts/subtree-sync.sh pull ejkernel   # reconcile a diverged mirror
```
