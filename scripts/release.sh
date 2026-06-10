#!/usr/bin/env bash
# Bump one workspace package's version and COMMIT — nothing leaves the machine.
# Keeps easydel's pins on the sibling libraries in sync and refreshes the lock.
# Publishing to GitHub/PyPI is a separate, explicit step: scripts/publish.sh.
#
#   scripts/release.sh <easydel|spectrax|ejkernel|eformer> <new-version> [--dry-run]
#
# Examples:
#   scripts/release.sh ejkernel 0.0.82          # bump + sync easydel pins + commit
#   scripts/release.sh easydel 0.4.1            # bump + refresh packaged README + commit
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
    echo "[dry-run] would: uv lock && git commit"
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

# 3. refresh the workspace lock and record the release locally
uv lock
git add "$pp" libs/easydel/pyproject.toml libs/easydel/README.md uv.lock 2>/dev/null || true
git commit -m "release: $dist v$version"

echo
echo "committed $dist v$version — publish later with:  scripts/publish.sh $lib"
