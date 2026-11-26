# Contributing to Eyvan

## Before you start

Eyvan is developed in a monorepo and mirrored here, so a patch to this
repository is applied upstream and mirrored back. Open an issue before a large
change; a rejected 2,000-line branch helps nobody.

## Running the tests

From a standalone clone, the lock is enough:

```bash
uv sync --extra dev
npm --prefix dashboard ci
npm --prefix dashboard run build   # Python tests also exercise serving the SPA
uv run pytest tests -m fast        # default quick-feedback smoke subset
# Configure disposable PostgreSQL as described below before the full gate:
EYVAN_REQUIRE_POSTGRES=1 uv run pytest tests -q -rs
```

The `fast` marker selects eight explicit modules in `tests/conftest.py`:
scheduler decisions and packing, configuration, provider boundaries, and
session-security regressions. This is smoke coverage, not a substitute for
focused tests of your change or the full suite.

The separate `slow` marker identifies integration-scale modules. Using
`-m "not slow"` gives broader coverage than `-m fast`, but remains a
multi-minute serial run; it is not the under-one-minute subset.

Nightly CI also runs the extended repository soak and isolated, tracked flaky
tests. See [testing and quarantine policy](docs/testing-policy.md) for marker
requirements and evidence limits. Quarantine never waives a failing full-suite
gate. To run the extended repository test locally:

```bash
uv run pytest tests/test_repository_soak.py --soak-iterations=1000 -q
```

The deployment-grade path installs the hash-pinned exports instead — what
CI, the Dockerfiles, and `eyvan head up` use:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --require-hashes -r requirements-docker.txt
uv pip install --python .venv/bin/python --require-hashes -r requirements-dev.txt
uv pip install --python .venv/bin/python --require-hashes -r requirements-build.txt
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .
EYVAN_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest tests
```

### Disposable PostgreSQL for the full gate

The full developer gate requires PostgreSQL; without that flag the suite can
skip PostgreSQL-specific tests while still exercising memory and SQLite.
Never use a production or persistent operator database: tests drop schemas.

With Docker Compose v2 and a running Docker daemon, use the separate ephemeral
service (loopback port 55432, not the persistent development service on 5432):

```bash
export POSTGRES_TEST_PASSWORD="$(openssl rand -hex 24)"
docker compose -f docker-compose.test.yml up -d --wait
export EYVAN_TEST_POSTGRES_URL="postgresql+asyncpg://postgres:${POSTGRES_TEST_PASSWORD}@127.0.0.1:55432/postgres"
EYVAN_REQUIRE_POSTGRES=1 uv run pytest tests -q -rs
# Always clean up, including after a failed test run:
docker compose -f docker-compose.test.yml down
unset EYVAN_TEST_POSTGRES_URL POSTGRES_TEST_PASSWORD
```

The database lives in tmpfs and is discarded on container removal. Run only
one suite against this service at a time: parallel independent suites share
its schema. Do not print resolved Compose configuration containing the password.

Without Docker, install PostgreSQL and set `EYVAN_TEST_POSTGRES_BINDIR` to its
`bin` directory (or put `initdb` and `pg_ctl` on PATH), then unset
`EYVAN_TEST_POSTGRES_URL`. The fixture starts and stops a private temporary
cluster on a free loopback port; `EYVAN_REQUIRE_POSTGRES=1` still makes missing
binaries or startup failure a failing gate, not a skip.

Dashboard:

```bash
npm --prefix dashboard ci && npm --prefix dashboard test
npm --prefix dashboard run build      # vitest does not type-check; the build does
```

Real-browser BFF qualification (after the Python dev install and dashboard build):

```bash
npm --prefix dashboard run browser:install  # downloads the pinned Chromium build
npm --prefix dashboard run test:browser -- --python "$PWD/.venv/bin/python"
```

On Linux, append `-- --with-deps` to `browser:install` to install browser system
libraries (may require elevated permissions); CI does this in its disposable
runner. Alternatively pass `--browser-executable /absolute/path/to/chrome` to
`test:browser` to use an already installed browser. Its measured version is
recorded in `browser-results/bff.json`, alongside assertions and failures.
The harness creates a fresh browser context, isolated in-memory HTTPS plane
and distinct-site fake IdP on ephemeral loopback ports. It never uses a saved
browser profile or real identity-provider account. The fixture accepts only
locally signed tokens; browser acceptance of its self-signed certificate is
intentional, not proof of deployed TLS or production JWKS configuration.
Both listeners and temporary certificate files are cleaned up on exit.
The blocking Python 3.11 CI lane runs this check and retains its JSON receipt;
local success is not evidence that hosted CI has run.

## What a good change looks like

- **Tests assert behaviour, not implementation.** Assert a persisted row, an
  emitted byte, an exit code, a returned value. A test that asserts a mock was
  called passes when the feature is broken.
- **Unknown is a state.** Never render an unreadable value as zero, empty, or
  healthy. If something cannot be determined, say so and keep it distinct.
- **No external I/O inside a database transaction.** Not a provider call, not an
  agent call, not an object-store read.
- **The scheduler stays pure.** `Scheduler.schedule` takes a snapshot and returns
  decisions; every decision is revalidated in a transaction before it applies.
- **A behaviour change to a wire format or an operator-visible default ships
  behind a flag that defaults to today's behaviour.**
- **Do not claim what you did not verify.** If a change is untested on hardware,
  say so in the commit message. Overstating evidence is the one thing that will
  get a patch rejected outright.

## Style

`ruff check` and `ruff format` must pass, line length 121. Google-style
docstrings. Import contracts are enforced: `uv run lint-imports` upstream, and
the layering is checked in CI.

Commit subjects are imperative and under 72 characters. No self-credit trailers.
