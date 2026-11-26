## Summary

<!-- What does this change and why? If it fixes an issue, link it. -->

## Evidence

<!-- What did you actually run? Paste the relevant results: focused tests, full suite outcome, lint/format, dashboard build. Claims without output are guesses. For hardware or cloud behavior, attach the receipt (`eyvan proof` where applicable). -->

```text
# paste results here
```

## Checklist

- [ ] Tests assert public behavior (persisted rows, exit codes, returned values), not mocks or log strings.
- [ ] New/changed behavior that is operator-visible or on the wire ships behind a flag that defaults to today's behavior.
- [ ] Docs updated where behavior changed (docs are tested by the suite — `tests/test_docs_are_grounded.py` and friends must pass).
- [ ] No new `EYVAN_*` env var without regenerating `docs/env-surface.md` (`python scripts/enumerate_env_surface.py`).
- [ ] AGPL-3.0 header present on every new `.py` / `.ts` / `.tsx` file (`python scripts/license_headers.py`).
- [ ] Dependency changes: `uv lock` re-run **and** `requirements-*.txt` exports regenerated.
- [ ] `uv run ruff check eyvan tests` and `uv run ruff format --check eyvan tests` pass.
- [ ] No external I/O inside database transactions; nothing that decides names a provider.
