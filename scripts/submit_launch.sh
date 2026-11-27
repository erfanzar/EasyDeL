#!/bin/bash
# Entrypoint wrapper for `ray job submit`: job drivers start in $HOME, but
# launch.py packages os.path.abspath(".") as the Ray working dir and resolves
# libs/ relatively, so it must run from the repo root.
cd "${EASYDEL_SRC:-$HOME/easydel-src}" || exit 1
exec "${EASYDEL_PYTHON:-$HOME/easy-venv/bin/python}" -u launch.py
