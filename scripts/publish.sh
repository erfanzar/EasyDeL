#!/usr/bin/env bash
# Publish one workspace package to GitHub + PyPI. This is the ONLY place a
# release leaves the machine — scripts/release.sh just bumps and commits.
#
#   scripts/publish.sh <easydel|spectrax|ejkernel|eformer> [--dry-run]
#
# Reads the package's current version, creates the <lib>-v<version> tag and
# pushes it to origin; the tag push triggers .github/workflows/publish.yaml,
# which builds the package and uploads it to PyPI.
#
# Note: this pushes the TAG only. Push the branch yourself (git push) so you
# stay in control of what lands on the remote branch.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

lib="${1:?usage: publish.sh <lib> [--dry-run]}"
dry="${2:-}"

declare -A DIST=([easydel]=easydel [spectrax]=spectrax-lib [ejkernel]=ejkernel [eformer]=eformer)
dist="${DIST[$lib]:?unknown lib: $lib (easydel|spectrax|ejkernel|eformer)}"
pp="libs/$lib/pyproject.toml"

version=$(grep -m1 '^version = ' "$pp" | sed 's/version = "\(.*\)"/\1/')
tag="$lib-v$version"

if [ "$dry" = "--dry-run" ]; then
    echo "[dry-run] would tag $tag and push it to origin (triggers PyPI publish for $dist $version)"
    exit 0
fi

if ! git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    git tag "$tag"
    echo "created tag $tag"
else
    echo "tag $tag already exists"
fi

git push origin "refs/tags/$tag"
echo "pushed $tag — CI is publishing $dist $version to PyPI"
