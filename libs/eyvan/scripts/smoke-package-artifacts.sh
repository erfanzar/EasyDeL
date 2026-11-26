#!/usr/bin/env bash
# Copyright (C) 2026 Erfan Zare Chavoshi
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly UV_BIN="${UV_BIN:-uv}"
command -v "${UV_BIN}" >/dev/null 2>&1 || { echo "missing required uv binary" >&2; exit 1; }
readonly PYTHON_BIN="$("${UV_BIN}" python find --no-python-downloads --no-project --system '>=3.11,<3.14')"
[[ -n "${PYTHON_BIN}" && "${PYTHON_BIN}" = /* && "${PYTHON_BIN}" != "${ROOT}/"* ]] || {
  echo "smoke Python must be a compatible absolute host interpreter outside the source tree" >&2
  exit 1
}
export UV_PYTHON_DOWNLOADS=never

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eyvan-package-smoke.XXXXXX")"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/dist" "${tmp}/outside-source"
build_venv="${tmp}/venv-build"
"${UV_BIN}" venv --no-python-downloads --python "${PYTHON_BIN}" "${build_venv}"
"${UV_BIN}" pip install --python "${build_venv}/bin/python" \
  --require-hashes -r "${ROOT}/requirements-build.txt"
VIRTUAL_ENV="${build_venv}" PATH="${build_venv}/bin:${PATH}" \
  "${UV_BIN}" build --no-build-isolation --wheel --sdist --out-dir "${tmp}/dist" "${ROOT}"

shopt -s nullglob
wheels=("${tmp}"/dist/*.whl)
sdists=("${tmp}"/dist/*.tar.gz)
[[ ${#wheels[@]} -eq 1 ]] || { echo "expected exactly one built wheel" >&2; exit 1; }
[[ ${#sdists[@]} -eq 1 ]] || { echo "expected exactly one built sdist" >&2; exit 1; }

smoke_artifact() {
  local kind="$1"
  local artifact="$2"
  local venv="${tmp}/venv-${kind}"
  "${UV_BIN}" venv --no-python-downloads --python "${PYTHON_BIN}" "${venv}"
  "${UV_BIN}" pip install --python "${venv}/bin/python" \
    --require-hashes -r "${ROOT}/requirements-build.txt"
  "${UV_BIN}" pip install --python "${venv}/bin/python" --no-build-isolation "${artifact}"
  "${UV_BIN}" pip uninstall --python "${venv}/bin/python" uv-build
  (
    cd "${tmp}/outside-source"
    unset PYTHONPATH
    "${venv}/bin/python" -I -c \
      'import pathlib, eyvan; path = pathlib.Path(eyvan.__file__).resolve(); assert "site-packages" in path.parts, path'
    for entrypoint in eyvan eyvan-server eyvan-agent eyvan-native-worker eyvan-migrate; do
      "${venv}/bin/${entrypoint}" --help >/dev/null
    done
  )
}

smoke_artifact wheel "${wheels[0]}"
smoke_artifact sdist "${sdists[0]}"
echo "Built wheel and sdist passed isolated installed-entrypoint smoke tests."
