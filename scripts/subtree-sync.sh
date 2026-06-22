#!/usr/bin/env bash
# Sync the libs/* subtrees with their standalone mirror repositories.
#
#   scripts/subtree-sync.sh push [lib...]   # monorepo -> mirrors (default: all four)
#   scripts/subtree-sync.sh pull [lib...]   # mirrors -> monorepo (squashed: 2 commits per lib)
#   scripts/subtree-sync.sh auto [lib...]   # push only the libs whose mirror is out of date
#                                           # (used by the pre-push git hook; DRY_RUN=1 to
#                                           # preview, SUBTREE_SYNC_SKIP=1 to bypass)
#
# `auto` runs from the pre-push hook so mirrors update whenever you `git push`
# and a library actually changed. CI (.github/workflows/sync-subtrees.yaml)
# remains the server-side backstop.
#
# Every commit pushed to a mirror has its message prefixed with ${MONO_SYNC_PREFIX}
# (default "easydel(mono-sync): ") so mirror history is clearly attributable to the
# monorepo sync. Per-commit granularity, author, committer and dates are preserved;
# only NEW commits (those not already on the mirror) are replayed, so existing mirror
# history is untouched and the push stays a fast-forward. Pulls are squashed so the
# monorepo history stays compact.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

declare -A REPOS=(
    [spectrax]="https://github.com/erfanzar/Spectrax"
    [ejkernel]="https://github.com/erfanzar/ejkernel"
    [eformer]="https://github.com/erfanzar/eformer"
    [eray]="https://github.com/erfanzar/eray"
)

MONO_SYNC_PREFIX="${MONO_SYNC_PREFIX:-easydel(mono-sync): }"

cmd="${1:-push}"
shift || true
libs=("$@")
[ ${#libs[@]} -eq 0 ] && libs=(spectrax ejkernel eformer eray)

if [ "$cmd" = "auto" ]; then
    # bypass requested
    if [ -n "${SUBTREE_SYNC_SKIP:-}" ]; then
        echo "subtree-sync: skipped (SUBTREE_SYNC_SKIP set)"
        exit 0
    fi
    # `git push` to a mirror fires this pre-push hook again -> guard against
    # infinite recursion, and only sync when the push target is the monorepo.
    if [ -n "${SUBTREE_SYNC_RUNNING:-}" ]; then
        exit 0
    fi
    if [ -n "${PRE_COMMIT_REMOTE_URL:-}" ] && [[ "${PRE_COMMIT_REMOTE_URL}" != *EasyDeL* ]]; then
        exit 0
    fi
    export SUBTREE_SYNC_RUNNING=1
fi

# Replay libs/$lib's subtree onto the mirror's current main, prefixing each NEW commit's
# message with ${MONO_SYNC_PREFIX} (idempotent: skips commits already prefixed). Preserves
# author/committer identity and dates. Echoes the head SHA to push, or "INSYNC" / "DIVERGED".
build_prefixed_head() {
    local lib="$1" remote_head="$2"
    local split_head split_tree remote_tree boundary new_parent c tree msg range
    if ! split_head=$(git subtree split --ignore-joins --prefix="libs/$lib" HEAD 2>/dev/null); then
        echo "failed to split libs/$lib from monorepo history" >&2
        return 1
    fi
    split_tree=$(git rev-parse "${split_head}^{tree}")

    if [ -n "$remote_head" ]; then
        if ! remote_tree=$(git rev-parse "${remote_head}^{tree}" 2>/dev/null); then
            echo "failed to read fetched mirror head for $lib: $remote_head" >&2
            return 1
        fi
        if [ "$split_tree" = "$remote_tree" ]; then echo INSYNC; return 0; fi
        # Boundary = newest split commit whose tree matches the mirror's current content.
        # Tree-based (not SHA-based) so it survives the message rewriting we apply below.
        boundary=""
        for c in $(git rev-list "$split_head"); do
            if [ "$(git rev-parse "${c}^{tree}")" = "$remote_tree" ]; then boundary="$c"; break; fi
        done
        if [ -z "$boundary" ]; then echo DIVERGED; return 0; fi
        range="${boundary}..${split_head}"
        new_parent="$remote_head"
    else
        range="$split_head"      # fresh mirror: replay the whole subtree history
        new_parent=""
    fi

    for c in $(git rev-list --reverse "$range"); do
        tree=$(git rev-parse "${c}^{tree}")
        msg=$(git log -1 --format=%B "$c")
        case "$msg" in "${MONO_SYNC_PREFIX}"*) : ;; *) msg="${MONO_SYNC_PREFIX}${msg}" ;; esac
        local pargs=()
        [ -n "$new_parent" ] && pargs=(-p "$new_parent")
        new_parent=$(
            GIT_AUTHOR_NAME="$(git log -1 --format=%an "$c")" \
            GIT_AUTHOR_EMAIL="$(git log -1 --format=%ae "$c")" \
            GIT_AUTHOR_DATE="$(git log -1 --format=%aI "$c")" \
            GIT_COMMITTER_NAME="$(git log -1 --format=%cn "$c")" \
            GIT_COMMITTER_EMAIL="$(git log -1 --format=%ce "$c")" \
            GIT_COMMITTER_DATE="$(git log -1 --format=%cI "$c")" \
            git commit-tree "$tree" "${pargs[@]}" -F - <<<"$msg"
        )
    done
    echo "$new_parent"
}

push_prefixed() {
    local lib="$1" repo="$2"
    local lsout remote_head remote_ref head
    if ! lsout=$(git ls-remote "$repo" main 2>/dev/null); then
        echo ">>> $lib: mirror unreachable, skipping (retry: scripts/subtree-sync.sh push $lib)"
        return 0
    fi
    remote_head=$(printf '%s' "$lsout" | cut -f1)
    if [ -n "$remote_head" ]; then
        remote_ref="refs/subtree-sync/$lib/main"
        if ! git fetch --quiet --no-tags --depth=1 "$repo" "+main:$remote_ref"; then
            echo ">>> $lib: failed to fetch mirror main from $repo" >&2
            return 1
        fi
        remote_head=$(git rev-parse "$remote_ref")
        git update-ref -d "$remote_ref" || true
    fi
    if ! head=$(build_prefixed_head "$lib" "$remote_head"); then
        echo ">>> $lib: failed to build mirror update" >&2
        return 1
    fi
    case "$head" in
        INSYNC) echo ">>> $lib: mirror up to date"; return 0 ;;
        DIVERGED)
            echo ">>> $lib: mirror history diverged from the monorepo subtree;" >&2
            echo "    reconcile with: scripts/subtree-sync.sh pull $lib" >&2
            return 1 ;;
    esac
    if [ -n "${DRY_RUN:-}" ]; then
        echo ">>> $lib: WOULD push ${head} -> $repo (main); new commit subjects:"
        if [ -n "$remote_head" ]; then
            git log --format='      %s' "${remote_head}..${head}"
        else
            git log --format='      %s' "$head"
        fi
        return 0
    fi
    echo ">>> $lib: pushing prefixed update -> $repo (main)"
    # Fully-qualify the destination: when the mirror is empty (fresh repo)
    # Git won't infer that `main` means refs/heads/main because the src is a
    # bare commit SHA, not a ref name. refs/heads/main works in both cases.
    git push "$repo" "${head}:refs/heads/main"
}

for lib in "${libs[@]}"; do
    repo="${REPOS[$lib]:?unknown lib: $lib}"
    case "$cmd" in
    push)
        echo ">>> libs/$lib -> $repo (main)"
        push_prefixed "$lib" "$repo"
        ;;
    pull)
        echo ">>> $repo (main) -> libs/$lib"
        git subtree pull --prefix="libs/$lib" "$repo" main --squash \
            -m "deps: sync libs/$lib from $repo"
        ;;
    auto)
        push_prefixed "$lib" "$repo"
        ;;
    *)
        echo "usage: $0 {push|pull|auto} [lib...]" >&2
        exit 1
        ;;
    esac
done
