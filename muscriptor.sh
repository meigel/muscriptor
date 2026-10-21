#!/usr/bin/env bash
# Convenience wrapper: run the muscriptor CLI from a working venv, without
# activating it first. Handy while developing when the project's dependencies
# live in a shared venv rather than one created by `uv sync`.
#
#   ./muscriptor.sh transcribe audio.wav --detect-key
#
# Override the venv with MUSCRIPTOR_VENV=/path/to/venv.
set -euo pipefail

VENV="${MUSCRIPTOR_VENV:-$HOME/work/venv/tinyTT}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "Error: no venv at $VENV (set MUSCRIPTOR_VENV)" >&2
    exit 1
fi

exec "$VENV/bin/python" -m muscriptor "$@"
