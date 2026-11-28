#!/usr/bin/env bash
# Bump one workspace package's version and COMMIT — nothing leaves the machine.
# Keeps easydel's pins on the sibling libraries in sync and refreshes the lock.
# Publishing to GitHub/PyPI is a separate, explicit step: scripts/publish.sh.
#
#   scripts/release.sh <easydel|spectrax|ejkernel|eformer|eray|surgecore|eyvan> <new-version> [--dry-run]
#
# Examples:
#   scripts/release.sh ejkernel 0.0.82          # bump + sync easydel pins + commit
#   scripts/release.sh easydel 0.4.1            # bump + refresh packaged README + commit
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

lib="${1:?usage: release.sh <lib> <version> [--dry-run]}"
version="${2:?usage: release.sh <lib> <version> [--dry-run]}"
dry="${3:-}"

declare -A DIST=([easydel]=easydel [spectrax]=spectrax-lib [ejkernel]=ejkernel [eformer]=eformer [eray]=eray [surgecore]=surgecore [eyvan]=eyvan)
dist="${DIST[$lib]:?unknown lib: $lib (easydel|spectrax|ejkernel|eformer|eray|surgecore|eyvan)}"
pp="libs/$lib/pyproject.toml"
[ -f "$pp" ] || { echo "missing $pp" >&2; exit 1; }

current=$(grep -m1 '^version = ' "$pp" | sed 's/version = "\(.*\)"/\1/')
echo ">>> $dist: $current -> $version"

dashboard_pkg="libs/$lib/dashboard/package.json"
dashboard_lock="libs/$lib/dashboard/package-lock.json"
helm_chart="libs/$lib/deploy/helm/$lib/Chart.yaml"

if [ "$dry" = "--dry-run" ]; then
    echo "[dry-run] would bump version in $pp"
    if [ "$lib" != "easydel" ]; then
        echo "[dry-run] would sync these pins in libs/easydel/pyproject.toml:"
        grep -nE "\"$dist(\[[^]]*\])?==" libs/easydel/pyproject.toml || echo "  (no pins found)"
    else
        echo "[dry-run] would refresh libs/easydel/README.md from root README.md"
    fi
    if [ -f "libs/$lib/Cargo.toml" ]; then
        echo "[dry-run] would bump [package] version in libs/$lib/Cargo.toml and the crate's entry in libs/$lib/Cargo.lock"
    fi
    if [ -f "$dashboard_pkg" ]; then
        echo "[dry-run] would bump version in $dashboard_pkg (and $dashboard_lock if present)"
    fi
    if [ -f "$helm_chart" ]; then
        echo "[dry-run] would bump version/appVersion in $helm_chart"
    fi
    echo "[dry-run] would: uv lock && git commit"
    exit 0
fi

[ -z "$(git status --porcelain)" ] || { echo "working tree not clean — commit or stash first" >&2; exit 1; }

# 1. bump the package version
sed -i "0,/^version = \"$current\"/s//version = \"$version\"/" "$pp"

# 1b. a Rust lib (one with libs/<lib>/Cargo.toml, e.g. surgecore) also carries its version in the crate's
# [package] table and in the crate's own entry of its Cargo.lock: bump both with sed, so no cargo is needed.
cargo_toml="libs/$lib/Cargo.toml"
cargo_lock="libs/$lib/Cargo.lock"
if [ -f "$cargo_toml" ]; then
    crate=$(sed -n '/^\[package\]/,/^\[/ s/^name = "\(.*\)"$/\1/p' "$cargo_toml" | head -n1)
    [ -n "$crate" ] || { echo "no [package] name in $cargo_toml" >&2; exit 1; }
    sed -i "/^\[package\]/,/^\[/ s/^version = \".*\"\$/version = \"$version\"/" "$cargo_toml"
    grep -qx "version = \"$version\"" "$cargo_toml" || { echo "failed to bump $cargo_toml" >&2; exit 1; }
    if [ -f "$cargo_lock" ]; then
        sed -i "/^name = \"$crate\"\$/{n;s/^version = \".*\"\$/version = \"$version\"/}" "$cargo_lock"
        grep -A1 -x "name = \"$crate\"" "$cargo_lock" | grep -x "version = \"$version\"" >/dev/null \
            || { echo "failed to bump $crate in $cargo_lock" >&2; exit 1; }
        echo "bumped crate $crate to $version in $cargo_toml and $cargo_lock"
    else
        echo "bumped crate $crate to $version in $cargo_toml (no $cargo_lock)"
    fi
fi

# 1c. packages with a companion dashboard or Helm chart (e.g. eyvan) keep their versions aligned
if [ -f "$dashboard_pkg" ]; then
    python3 -c '
import json, pathlib, sys
ver = sys.argv[1]
for p in [pathlib.Path(x) for x in sys.argv[2:] if pathlib.Path(x).is_file()]:
    data = json.loads(p.read_text(encoding="utf-8"))
    data["version"] = ver
    if isinstance(data.get("packages"), dict) and "" in data["packages"]:
        data["packages"][""]["version"] = ver
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
' "$version" "$dashboard_pkg" "$dashboard_lock"
    echo "bumped dashboard version to $version"
fi
if [ -f "$helm_chart" ]; then
    python3 -c '
import pathlib, re, sys
ver, p = sys.argv[1], pathlib.Path(sys.argv[2])
text = p.read_text(encoding="utf-8")
text = re.sub(r"^version:\s*.*$", f"version: {ver}", text, count=1, flags=re.M)
text = re.sub(r"^appVersion:\s*.*$", f"appVersion: \"{ver}\"", text, count=1, flags=re.M)
p.write_text(text, encoding="utf-8")
' "$version" "$helm_chart"
    echo "bumped Helm chart version/appVersion to $version in $helm_chart"
fi

# 2. keep easydel's pins on the sibling libraries in lockstep with the release
if [ "$lib" != "easydel" ]; then
    if grep -qE "\"$dist(\[[^]]*\])?==" libs/easydel/pyproject.toml; then
        # Replace only the version token (up to a space/semicolon/quote), so any
        # trailing environment marker like ` ; sys_platform == 'linux'` on the pin
        # survives the bump instead of being swallowed by the old `[^"]+`.
        sed -i -E "s/(\"$dist(\[[^]]*\])?==)[^\"; ]+/\1$version/g" libs/easydel/pyproject.toml
        echo "synced easydel pins:"
        grep -nE "\"$dist(\[[^]]*\])?==" libs/easydel/pyproject.toml
    else
        echo "no pins for $dist in libs/easydel/pyproject.toml (skipping)"
    fi
else
    # the packaged README is a copy of the root README — refresh it at release
    cp README.md libs/easydel/README.md
fi

# 3. refresh the workspace lock and record the release locally
uv lock
git add "$pp" libs/easydel/pyproject.toml libs/easydel/README.md uv.lock 2>/dev/null || true
for f in "$cargo_toml" "$cargo_lock" "$dashboard_pkg" "$dashboard_lock" "$helm_chart"; do
    if [ -f "$f" ]; then git add "$f"; fi
done
git commit -m "release: $dist v$version"

echo
echo "committed $dist v$version — publish later with:  scripts/publish.sh $lib"
