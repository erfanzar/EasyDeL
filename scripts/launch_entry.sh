#!/usr/bin/env bash
# Entrypoint for `ray job submit`: job drivers start in $HOME, but launch.py
# must run from the repo root so its ray.init packages the right working dir.
set -euo pipefail
cd "${EASYDEL_SRC:-$HOME/easydel-src}"
exec "${EASYDEL_PYTHON:-$HOME/easy-venv/bin/python}" -u launch.py
