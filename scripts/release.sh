#!/usr/bin/env bash
# Release one workspace package: bump its version, keep easydel's pins on the
# sibling libraries in sync, refresh the lock, commit, and tag. Publishing to
# PyPI happens in CI when the tag is pushed (.github/workflows/publish.yaml).
#
#   scripts/release.sh <easydel|spectrax|ejkernel|eformer> <new-version> [--dry-run]
#
# Examples:
#   scripts/release.sh ejkernel 0.0.81          # bump + sync easydel pins + tag ejkernel-v0.0.81
#   scripts/release.sh easydel 0.4.1            # bump + refresh packaged README + tag easydel-v0.4.1
#
# After the script: `git push origin <branch> <tag>` triggers the publish.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

lib="${1:?usage: release.sh <lib> <version> [--dry-run]}"
version="${2:?usage: release.sh <lib> <version> [--dry-run]}"
dry="${3:-}"

declare -A DIST=([easydel]=easydel [spectrax]=spectrax-lib [ejkernel]=ejkernel [eformer]=eformer)
dist="${DIST[$lib]:?unknown lib: $lib (easydel|spectrax|ejkernel|eformer)}"
pp="libs/$lib/pyproject.toml"
[ -f "$pp" ] || { echo "missing $pp" >&2; exit 1; }

current=$(grep -m1 '^version = ' "$pp" | sed 's/version = "\(.*\)"/\1/')
echo ">>> $dist: $current -> $version"

if [ "$dry" = "--dry-run" ]; then
    echo "[dry-run] would bump version in $pp"
    if [ "$lib" != "easydel" ]; then
        echo "[dry-run] would sync these pins in libs/easydel/pyproject.toml:"
        grep -nE "\"$dist(\[[^]]*\])?==" libs/easydel/pyproject.toml || echo "  (no pins found!)"
    else
        echo "[dry-run] would refresh libs/easydel/README.md from root README.md"
    fi
    echo "[dry-run] would: uv lock && git commit && git tag $lib-v$version && uv build --package $dist"
    exit 0
fi

[ -z "$(git status --porcelain)" ] || { echo "working tree not clean — commit or stash first" >&2; exit 1; }

# 1. bump the package version
sed -i "0,/^version = \"$current\"/s//version = \"$version\"/" "$pp"

# 2. keep easydel's pins on the sibling libraries in lockstep with the release
if [ "$lib" != "easydel" ]; then
    sed -i -E "s/\"$dist(\[[^]]*\])?==[^\"]+\"/\"$dist\1==$version\"/g" libs/easydel/pyproject.toml
    echo "synced easydel pins:"
    grep -nE "\"$dist(\[[^]]*\])?==" libs/easydel/pyproject.toml
else
    # the packaged README is a copy of the root README — refresh it at release
    cp README.md libs/easydel/README.md
fi

# 3. refresh the workspace lock and record the release
uv lock
git add "$pp" libs/easydel/pyproject.toml libs/easydel/README.md uv.lock 2>/dev/null || true
git commit -m "release: $dist v$version"
git tag "$lib-v$version"

# 4. build artifacts locally as a sanity check (CI rebuilds + publishes on tag push)
uv build --package "$dist" --out-dir dist/
echo
echo "tagged $lib-v$version — push with:  git push origin HEAD $lib-v$version"
