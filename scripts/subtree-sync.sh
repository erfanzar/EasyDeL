#!/usr/bin/env bash
# Sync the libs/* subtrees with their standalone mirror repositories.
#
#   scripts/subtree-sync.sh push [lib...]   # monorepo -> mirrors (default: all three)
#   scripts/subtree-sync.sh pull [lib...]   # mirrors -> monorepo (squashed: 2 commits per lib)
#
# CI does the push automatically on every push to main/vnext that touches
# libs/ (see .github/workflows/sync-subtrees.yaml); this script is the manual
# fallback and the way to pull in changes that landed on a mirror directly.
# Pushed mirror commits keep per-commit messages; pulls are squashed so the
# monorepo history stays compact.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

declare -A REPOS=(
    [spectrax]="https://github.com/erfanzar/Spectrax"
    [ejkernel]="https://github.com/erfanzar/ejkernel"
    [eformer]="https://github.com/erfanzar/eformer"
)

cmd="${1:-push}"
shift || true
libs=("$@")
[ ${#libs[@]} -eq 0 ] && libs=(spectrax ejkernel eformer)

for lib in "${libs[@]}"; do
    repo="${REPOS[$lib]:?unknown lib: $lib}"
    case "$cmd" in
    push)
        echo ">>> libs/$lib -> $repo (main)"
        git subtree push --prefix="libs/$lib" "$repo" main
        ;;
    pull)
        echo ">>> $repo (main) -> libs/$lib"
        git subtree pull --prefix="libs/$lib" "$repo" main --squash \
            -m "deps: sync libs/$lib from $repo"
        ;;
    *)
        echo "usage: $0 {push|pull} [lib...]" >&2
        exit 1
        ;;
    esac
done
