#!/usr/bin/env bash
# Sync the libs/* subtrees with their standalone mirror repositories.
#
#   scripts/subtree-sync.sh push [lib...]   # monorepo -> mirrors (default: all three)
#   scripts/subtree-sync.sh pull [lib...]   # mirrors -> monorepo (squashed: 2 commits per lib)
#   scripts/subtree-sync.sh auto [lib...]   # push only the libs whose mirror is out of date
#                                           # (used by the pre-push git hook; DRY_RUN=1 to
#                                           # preview, SUBTREE_SYNC_SKIP=1 to bypass)
#
# `auto` runs from the pre-push hook so mirrors update whenever you `git push`
# and a library actually changed. CI (.github/workflows/sync-subtrees.yaml)
# remains the server-side backstop. Pushed mirror commits keep per-commit
# messages; pulls are squashed so the monorepo history stays compact.
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

if [ "$cmd" = "auto" ] && [ -n "${SUBTREE_SYNC_SKIP:-}" ]; then
    echo "subtree-sync: skipped (SUBTREE_SYNC_SKIP set)"
    exit 0
fi

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
    auto)
        # mirror unreachable -> warn and continue (never block a push on a
        # mirror outage); mirror up to date -> skip; otherwise push (a real
        # divergence makes `git subtree push` fail loudly -> reconcile with
        # `scripts/subtree-sync.sh pull <lib>`)
        if ! remote_head=$(git ls-remote "$repo" main 2>/dev/null | cut -f1) || [ -z "$remote_head" ]; then
            echo ">>> $lib: mirror unreachable, skipping (sync later with: scripts/subtree-sync.sh push $lib)"
            continue
        fi
        split_head=$(git subtree split --prefix="libs/$lib" HEAD 2>/dev/null)
        if [ "$split_head" = "$remote_head" ]; then
            echo ">>> $lib: mirror up to date"
            continue
        fi
        if [ -n "${DRY_RUN:-}" ]; then
            echo ">>> $lib: WOULD push (local $split_head != mirror $remote_head)"
            continue
        fi
        echo ">>> $lib: pushing update -> $repo (main)"
        git subtree push --prefix="libs/$lib" "$repo" main
        ;;
    *)
        echo "usage: $0 {push|pull|auto} [lib...]" >&2
        exit 1
        ;;
    esac
done
