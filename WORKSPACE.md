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

```md
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

Releasing is split into two explicit steps — nothing leaves the machine
until you run the second one:

```bash
scripts/release.sh ejkernel 0.0.82   # bump + sync easydel pins + lock + COMMIT (local only)
scripts/publish.sh ejkernel          # tag ejkernel-v0.0.82 + push the tag -> CI publishes to PyPI
```

`.github/workflows/publish.yaml` builds and publishes exactly the tagged
package (PyPI trusted publishing, or the `PYPI_API_TOKEN` secret).
`publish.sh` pushes the tag only — push the branch yourself with `git push`.

## Mirror sync

Mirrors update automatically when you `git push`: the pre-push hook
(`scripts/subtree-sync.sh auto`) detects which of spectrax/ejkernel/eformer
changed and subtree-pushes just those to their standalone repos
(`SUBTREE_SYNC_SKIP=1 git push` to bypass; mirror outages never block the
push). `.github/workflows/sync-subtrees.yaml` is the server-side backstop
(requires the `SUBTREE_SYNC_TOKEN` secret: fine-grained PAT, Contents
read/write on the three mirrors). Manual fallback:

```bash
scripts/subtree-sync.sh push            # all three
scripts/subtree-sync.sh pull ejkernel   # reconcile a diverged mirror
```

## Dev tooling

`uv sync` installs the `dev` group by default: `pytest`, `pre-commit`,
`ruff`, `import-linter`, `basedpyright`. Activate the git hooks once per
clone:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```
