#!/usr/bin/env bash
# Convenience wrapper: activate the working venv and run muscriptor CLI.
set -euo pipefail

VENV="$HOME/work/venv/tinyTT"
if [ ! -d "$VENV" ]; then
    echo "Error: venv not found at $VENV" >&2
    exit 1
fi

source "$VENV/bin/activate"
exec python -m muscriptor "$@"
