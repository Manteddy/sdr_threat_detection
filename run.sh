#!/usr/bin/env bash
#
# Launcher for the FPV drone detector GUI.
#
# Runs the venv's Python directly so no `source .venv/bin/activate`
# dance is needed. `cd`s into the repo first so engine_config.yaml is
# found regardless of where you invoke this script from.
#
# First-time setup: create the venv + install deps once with
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install numpy pyqtgraph PyQt5 pyyaml
# (or pip install -r requirements.txt on Linux with libiio installed).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
    echo "error: .venv/bin/python not found in $REPO_DIR" >&2
    echo "       run the one-time setup first:" >&2
    echo "         python3 -m venv .venv" >&2
    echo "         source .venv/bin/activate" >&2
    echo "         pip install numpy pyqtgraph PyQt5 pyyaml" >&2
    echo "       (or pip install -r requirements.txt on a Pluto-attached Linux host)" >&2
    exit 1
fi

exec ./.venv/bin/python drone_detector_enhanced.py "$@"
